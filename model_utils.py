import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import cv2
import numpy as np

MODEL_DIR = 'model'
WEIGHTS_DIR = 'weights'
# Check both local weights dir and external_repo/weights if possible, 
# but for now let's stick to a standard path or allow configuration.
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'modeldense1.h5')

def ensure_directories():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

# Removed automatic download to rely on local files as requested

def create_model():
    """Recreates the DenseNet201 architecture used in the external repo"""
    # Note: The external repo used 'pooling=max' and 'weights=imagenet'
    conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, pooling='max', weights='imagenet')
    
    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='softmax'))
    
    # Freeze layers as per original code (partial freeze)
    train_layers = [layer for layer in conv_base.layers[::-1][:5]]
    for layer in conv_base.layers:
        if layer not in train_layers:
            layer.trainable = False
            
    return model

def load_cancer_model():
    """Loads the model and weights"""
    ensure_directories()
    # download_weights_if_needed() - Removed as per user request
    
    try:
        # Strategy: Create structure then load weights
        # The original repo did model.save('model/model.h5') which saves architecture+weights
        # But also had a separate weights download. 
        # We will try to build the model structure and load the downloaded weights.
        
        # If the full model file exists, we could try loading that, 
        # but creating fresh and loading weights is often more robust across TF versions.
        if os.path.exists(WEIGHTS_PATH):
            try:
                print(f"Attempting to load full model from {WEIGHTS_PATH}...")
                model = tf.keras.models.load_model(WEIGHTS_PATH)
                print("Model loaded successfully using load_model.")
                model.weights_loaded = True
                return model
            except Exception as e:
                print(f"load_model failed ({e}), falling back to create_model + load_weights...")
        
        # Fallback: Create structure then load weights
        model = create_model()
        # Compile to avoid warnings
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"], loss='categorical_crossentropy')
        
        if os.path.exists(WEIGHTS_PATH):
            try:
                model.load_weights(WEIGHTS_PATH)
                print("Model weights loaded successfully.")
                model.weights_loaded = True
                return model
            except Exception as e:
                print(f"Error loading weights: {e}")
                model.weights_loaded = False
        else:
            print("Warning: Weights file not found. Returning model with ImageNet initialization.")
            model.weights_loaded = False
            
        return model

    except Exception as e:
        print(f"Critical error creating model: {e}")
        # Even in critical error, try to return a basic uncompiled model if possible, or None as last resort
        try:
            model = create_model()
            model.weights_loaded = False
            return model
        except:
            return None

def preprocess_image(image_file):
    """
    Preprocesses the image for the model.
    Includes CLAHE for contrast enhancement and standard DenseNet normalization.
    """
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Read image using cv2/numpy
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Failed to decode image.")
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Convert to LAB for better luminance-specific equalization
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 2. Sharpening (preserving original logic but making it subtler)
        kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])
        img = cv2.filter2D(img, -1, kernel)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # 3. DenseNet specific normalization
        # Note: img is currently 0-255 uint8, preprocess_input expects 0-255 float
        img = img.astype('float32')
        img = densenet_preprocess(img)
        
        # Reshape for model input (batch size 1)
        img_reshape = img.reshape(-1, 224, 224, 3)
        
        return img_reshape
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

CLASS_NAMES = [
    'Benign with Density=1',
    'Malignant with Density=1',
    'Benign with Density=2',
    'Malignant with Density=2',
    'Benign with Density=3',
    'Malignant with Density=3',
    'Benign with Density=4',
    'Malignant with Density=4'
]

# Minimum softmax confidence required to accept an image as a valid mammogram scan.
# The model outputs 8 classes via softmax. For a random/unrelated image the
# probabilities are spread evenly (~12.5% each). A genuine mammogram will have
# one class dominate. We require at least 50% confidence to accept the result.
CONFIDENCE_THRESHOLD = 0.50

def get_prediction_result(predictions):
    """
    Parses the prediction array into a meaningful result.
    Groups probabilities of all benign classes and all malignant classes to
    obtain a robust "Diagnosis Confidence".
    """
    try:
        # Handle predictions (ensure it's a flat array)
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        if np.any(np.isnan(predictions)):
            return "Error", 0.0, "Unknown", {}

        # 1. Identify specific density/class for reporting
        pred_idx = np.argmax(predictions)
        top_raw_score = float(predictions[pred_idx])
        result_str = CLASS_NAMES[pred_idx]
        
        # Parse density
        density = "Unknown"
        if 'Density=' in result_str:
            density = result_str.split('Density=')[1]

        # 2. Group probabilities for Diagnosis Confidence
        # CLASS_NAMES map: 0,2,4,6 are Benign; 1,3,5,7 are Malignant
        benign_indices = [0, 2, 4, 6]
        malignant_indices = [1, 3, 5, 7]
        
        benign_prob = sum(predictions[i] for i in benign_indices)
        malignant_prob = sum(predictions[i] for i in malignant_indices)
        
        # Determine diagnosis and the "Grouped Confidence"
        if benign_prob >= malignant_prob:
            diagnosis = "Benign"
            diagnosis_confidence = float(benign_prob)
        else:
            diagnosis = "Malignant"
            diagnosis_confidence = float(malignant_prob)

        # 3. Confidence gate
        # We check diagnosis_confidence because the split between fine-grained 
        # density classes can be ambiguous, but the Benign/Malignant split is 
        # what matters most for the diagnosis.
        if diagnosis_confidence < CONFIDENCE_THRESHOLD:
            return "NotMammogram", diagnosis_confidence, "Unknown", {}

        full_results = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        # Add summary grouped scores to metadata
        full_results["_GroupedBenign"] = float(benign_prob)
        full_results["_GroupedMalignant"] = float(malignant_prob)

        return diagnosis, diagnosis_confidence, density, full_results

    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return "Error", 0.0, "Unknown", {}
