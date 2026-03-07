import os
import numpy as np
import cv2
import model_utils
from io import BytesIO

def test_confidence_logic():
    print("Testing Confidence Improvement Logic...")
    
    # Simulate an 8-class prediction where probabilities are split
    # Index: 0=B1, 1=M1, 2=B2, 3=M2, 4=B3, 5=M3, 6=B4, 7=M4
    # Example: Clear Benign case but split across densities
    # B1=0.4, B2=0.3, M1=0.1, others=0.05
    test_preds = np.array([0.4, 0.1, 0.3, 0.05, 0.05, 0.05, 0.025, 0.025])
    
    # Old logic would have taken max: 0.4 (40%) and likely rejected it or shown low confidence.
    # New logic should sum Benign (0,2,4,6) = 0.4 + 0.3 + 0.05 + 0.025 = 0.775 (77.5%)
    
    diagnosis, confidence, density, full_results = model_utils.get_prediction_result(test_preds)
    
    print(f"Test 1 (Grouped Benign):")
    print(f"  Diagnosis: {diagnosis}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Expected Confidence: ~0.775")
    
    assert diagnosis == "Benign"
    assert confidence > 0.7
    
    # Example 2: Malignant case split
    test_preds_m = np.array([0.1, 0.45, 0.05, 0.3, 0.025, 0.05, 0.015, 0.01])
    # Malignant (1,3,5,7) = 0.45 + 0.3 + 0.05 + 0.01 = 0.81
    
    diagnosis_m, confidence_m, density_m, full_results_m = model_utils.get_prediction_result(test_preds_m)
    
    print(f"Test 2 (Grouped Malignant):")
    print(f"  Diagnosis: {diagnosis_m}")
    print(f"  Confidence: {confidence_m:.4f}")
    print(f"  Expected Confidence: ~0.81")
    
    assert diagnosis_m == "Malignant"
    assert confidence_m > 0.8
    
    print("\nCheck Successful! Grouped confidence logic is working as intended.")

def test_preprocessing():
    print("\nTesting Preprocessing (CLAHE)...")
    # Create a dummy image
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (150, 150), 50, (128, 128, 128), -1) # Mid-gray circle
    
    _, buffer = cv2.imencode('.jpg', dummy_img)
    image_file = BytesIO(buffer)
    
    processed = model_utils.preprocess_image(image_file)
    
    print(f"Processed shape: {processed.shape}")
    assert processed.shape == (1, 224, 224, 3)
    # Check if range is roughly standardized (not just 0-1)
    # densenet_preprocess subtracts mean and scales
    print(f"Min value: {processed.min():.4f}")
    print(f"Max value: {processed.max():.4f}")
    
    print("Preprocessing check successful!")

if __name__ == "__main__":
    test_confidence_logic()
    test_preprocessing()
