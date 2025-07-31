import requests
import json
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Health check failed: {e}")

def test_labels():
    """Test the labels endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/labels")
        print("Available Labels:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Labels check failed: {e}")

def test_prediction(audio_file_path, threshold=0.6):
    """Test the prediction endpoint"""
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {'threshold': threshold}
            response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
            
            print(f"Prediction for {audio_file_path}:")
            print(json.dumps(response.json(), indent=2))
            print()
    except Exception as e:
        print(f"Prediction failed: {e}")

def main():
    """Run all tests"""
    print("Testing Akan Profanity Detection API")
    print("=" * 40)
    
    # Test health endpoint
    test_health()
    
    # Test labels endpoint
    test_labels()
    
    # Test prediction (if audio file exists)
    audio_file = "test_audio.wav"  # Replace with your test audio file
    if os.path.exists(audio_file):
        test_prediction(audio_file)
    else:
        print(f"Test audio file '{audio_file}' not found. Skipping prediction test.")
        print("To test prediction, place an audio file named 'test_audio.wav' in the current directory.")

if __name__ == "__main__":
    main() 