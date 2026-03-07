import os
import tensorflow as tf
import numpy as np
import cv2
import model_utils

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_model():
    print("1. Loading model...")
    model = model_utils.load_cancer_model()
    if model is None:
        print("FAILED: Model could not be loaded.")
        return

    print("2. Model loaded. Verifying weights...")
    # Check if we can run a dummy prediction
    try:
        # Load test image
        img_path = 'test_image.jpg'
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found. Creating dummy image.")
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(img_path, dummy_img)
        
        print(f"3. Processing {img_path}...")
        with open(img_path, 'rb') as f:
            processed_img = model_utils.preprocess_image(f)
            
        print("4. Running prediction...")
        preds = model.predict(processed_img, verbose=0)[0]
        print(f"   Raw predictions: {preds}")
        
        result, confidence, density, details = model_utils.get_prediction_result(preds)
        
        print("\n=== RESULTS ===")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Density: {density}")
        print(f"Details: {details}")
        
        if confidence < 0.5:
             print("\nNote: Confidence is low. This is expected for a dummy/random image, but ensures pipeline works.")
        else:
             print("\nSuccess: Model returned a confident prediction.")
             
    except Exception as e:
        print(f"FAILED: Error during prediction: {e}")
        with open("verification_result.txt", "w") as f:
            f.write(f"FAILED: Error during prediction: {e}\n")

if __name__ == "__main__":
    # Redirect stdout to file for easier reading
    import sys
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self
        def __del__(self):
            sys.stdout = self.stdout
            self.file.close()
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    sys.stdout = Tee('verification_result.txt', 'w')
    test_model()
