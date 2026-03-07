
import requests
import numpy as np
import cv2
import os

# Create a dummy image
img = np.zeros((224, 224, 3), dtype=np.uint8)
# Add some "features" (random noise) to avoid completely blank input if that matters
img[:] = np.random.randint(0, 256, (224, 224, 3))
cv2.imwrite('test_image.jpg', img)

url = 'http://127.0.0.1:5000/predict-image'
files = {'file': open('test_image.jpg', 'rb')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)
