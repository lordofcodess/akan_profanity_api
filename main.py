import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import tempfile
import os
from pydantic import BaseModel
from typing import Optional, List

# Initialize FastAPI app
app = FastAPI(
    title="Akan Profanity Detection API",
    description="API for detecting profanity in Akan language audio files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your trained labels — same order as training
commands = np.array(['Profane', 'non_profane'])

# Load the model
model = None

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        model = tf.keras.models.load_model("profanity.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception("Failed to load model")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

def get_spectrogram_librosa(y, sr=16000):
    """Generate spectrogram from audio waveform"""
    # Ensure waveform is exactly 1 second
    target_len = 16000
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Convert to tensor and pad
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_spectrogram_stft(y, sr=16000):
    """Generate spectrogram from audio waveform for streaming analysis"""
    # Pad or trim to exactly 16000 samples
    target_len = 16000
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)  # [124, 129]
    return spectrogram

class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    is_profane: bool
    is_uncertain: bool

class SegmentDetection(BaseModel):
    start_time: float
    end_time: float
    detected_label: str
    confidence: float
    is_profane: bool
    is_uncertain: bool

class StreamingPredictionResponse(BaseModel):
    total_duration: float
    segments_analyzed: int
    profane_segments: List[SegmentDetection]
    clean_segments: List[SegmentDetection]
    summary: str

def predict_profanity(audio_data: bytes, threshold: float = 0.6) -> PredictionResponse:
    """Predict profanity from audio data"""
    try:
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Load and preprocess audio
        y, sr = librosa.load(temp_file_path, sr=16000)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = librosa.util.normalize(y)

        # Generate spectrogram
        spectrogram = get_spectrogram_librosa(y)

        # Prepare input for model
        input_tensor = tf.expand_dims(spectrogram, axis=0)  # [1, time, freq]
        input_tensor = tf.expand_dims(input_tensor, axis=-1)  # [1, time, freq, 1]

        # Predict
        predictions = model(input_tensor)
        probs = tf.nn.softmax(predictions[0])
        predicted_index = tf.argmax(probs).numpy()
        confidence = tf.reduce_max(probs).numpy()
        predicted_label = commands[predicted_index]

        # Clean up temporary file
        os.unlink(temp_file_path)

        # Determine if it's profane (binary classification)
        is_profane = predicted_label == 'Profane' and confidence >= threshold
        
        # Determine if prediction is uncertain (low confidence)
        is_uncertain = confidence < threshold

        return PredictionResponse(
            predicted_label=predicted_label,
            confidence=float(confidence),
            is_profane=is_profane,
            is_uncertain=is_uncertain
        )

    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

def predict_profanity_stream(audio_data: bytes, threshold: float = 0.6, sr: int = 16000, window_sec: float = 1.0, hop_sec: float = 0.5) -> StreamingPredictionResponse:
    """Predict profanity from long audio data using streaming analysis"""
    try:
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Load and preprocess audio
        y, sr = librosa.load(temp_file_path, sr=sr)
        y = librosa.util.normalize(y)
        duration = len(y) / sr

        win_len = int(window_sec * sr)
        hop_len = int(hop_sec * sr)
        segments_analyzed = (len(y) - win_len) // hop_len + 1

        profane_segments = []
        clean_segments = []

        for start in range(0, len(y) - win_len + 1, hop_len):
            end = start + win_len
            segment = y[start:end]
            spec = get_spectrogram_stft(segment)
            spec = tf.expand_dims(spec, axis=-1)  # [124,129,1]
            spec = tf.expand_dims(spec, axis=0)   # [1,124,129,1]

            prediction = model(spec)
            probs = tf.nn.softmax(prediction[0]).numpy()
            probs = np.clip(probs, 0, 1)
            pred_index = np.argmax(probs)
            confidence = float(probs[pred_index])
            label = commands[pred_index]

            start_time = start / sr
            end_time = end / sr

            segment_detection = SegmentDetection(
                start_time=start_time,
                end_time=end_time,
                detected_label=label,
                confidence=confidence,
                is_profane=confidence > threshold and label == "Profane",
                is_uncertain=confidence < threshold
            )

            if segment_detection.is_profane:
                profane_segments.append(segment_detection)
            else:
                clean_segments.append(segment_detection)

        # Clean up temporary file
        os.unlink(temp_file_path)

        # Generate summary
        if profane_segments:
            summary = f"⚠️ Found {len(profane_segments)} profane segments out of {segments_analyzed} analyzed"
        else:
            summary = f"✅ No profanity detected in {segments_analyzed} analyzed segments"

        return StreamingPredictionResponse(
            total_duration=duration,
            segments_analyzed=segments_analyzed,
            profane_segments=profane_segments,
            clean_segments=clean_segments,
            summary=summary
        )

    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Akan Profanity Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_profanity_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.6
):
    """
    Predict profanity in uploaded audio file (for short audio files)
    
    - **file**: Audio file (WAV, MP3, etc.)
    - **threshold**: Confidence threshold (default: 0.6)
    """
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Read file content
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Make prediction
    result = predict_profanity(audio_data, threshold)
    return result

@app.post("/predict-stream", response_model=StreamingPredictionResponse)
async def predict_profanity_stream_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.6,
    window_sec: Optional[float] = 1.0,
    hop_sec: Optional[float] = 0.5
):
    """
    Predict profanity in uploaded audio file using streaming analysis (for long audio files)
    
    - **file**: Audio file (WAV, MP3, etc.)
    - **threshold**: Confidence threshold (default: 0.6)
    - **window_sec**: Analysis window size in seconds (default: 1.0)
    - **hop_sec**: Hop size between windows in seconds (default: 0.5)
    """
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Read file content
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Make streaming prediction
    result = predict_profanity_stream(audio_data, threshold, window_sec=window_sec, hop_sec=hop_sec)
    return result

@app.get("/labels")
async def get_labels():
    """Get available prediction labels"""
    return {"labels": commands.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 