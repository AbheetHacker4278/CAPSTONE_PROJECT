import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_local_bridge():
    """
    Tests if the main app can communicate with the model api locally.
    1. Run 'python model_api.py' in one terminal.
    2. Run this script in another.
    """
    url = "http://localhost:7860" # Default port in model_api.py
    test_image = "test_image.jpg" # Ensure this exists
    
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found. Please place an image named test_image.jpg in this folder.")
        return

    print(f"Connecting to Model API at {url}...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{url}/predict", files=files)
            
        if response.status_code == 200:
            print("✅ Success! Model API returned:")
            print(response.json())
        else:
            print(f"❌ Failed. Status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")

if __name__ == "__main__":
    test_local_bridge()
