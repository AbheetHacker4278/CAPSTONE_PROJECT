import os
import logging
from flask import Flask, request, jsonify
import numpy as np
import model_utils
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the heavy model once at startup
logger.info("Loading heavy DenseNet201 model...")
try:
    image_model = model_utils.load_cancer_model()
    if image_model is not None:
        logger.info("Model loaded successfully.")
    else:
        logger.error("Failed to load model architecture/weights.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    image_model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": image_model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if image_model is None:
        return jsonify({"error": "Model not loaded on server"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # Preprocess using existing utility
        processed_img = model_utils.preprocess_image(file)
        if processed_img is None:
            return jsonify({"error": "Failed to process image"}), 400

        # Run inference
        predictions = image_model.predict(processed_img)[0]
        logger.info(f"Predictions generated: {predictions}")

        # Parse results using existing utility
        result, confidence, density, full_results = model_utils.get_prediction_result(predictions)

        return jsonify({
            "status": "success",
            "prediction": result,
            "confidence": float(confidence),
            "density": density,
            "details": full_results
        })

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Listen on all interfaces for container/remote deployment
    # Hugging Face Spaces defaults to 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
