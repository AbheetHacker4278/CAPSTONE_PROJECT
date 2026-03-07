import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import warnings
import logging
import base64
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
# Import our new model utility
import model_utils

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

# Configure logging so errors are visible in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API setup ──────────────────────────────────────────────────────────
# Key MUST be provided in .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    # You might want to raise an error or handle this gracefully
    # raise ValueError("Missing GEMINI_API_KEY")

GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]
genai.configure(api_key=GEMINI_API_KEY)

def _get_gemini_model():
    """Returns the first available Gemini model instance."""
    for name in GEMINI_MODELS:
        try:
            return genai.GenerativeModel(name), name
        except Exception:
            continue
    return None, None

def is_mammogram_image(image_bytes: bytes) -> tuple:
    """
    Uses Gemini to verify whether the uploaded image is a mammogram
    or breast histology/ultrasound image suitable for cancer analysis.

    Returns:
        (True, "")           – image is a valid medical breast scan
        (False, reason_str)  – image is NOT a mammogram; reason_str explains why
    """
    # Detect MIME type from magic bytes
    if image_bytes[:3] == b'\xff\xd8\xff':
        mime = "image/jpeg"
    elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    else:
        mime = "image/jpeg"  # safe default

    prompt = (
        "You are a medical image classifier. "
        "Look at this image and answer ONLY with one of these two responses:\n"
        "YES - if the image is a mammogram, breast ultrasound, or breast histology/pathology slide.\n"
        "NO: <reason> - if the image is anything else (e.g. a selfie, landscape, X-ray of another body part, etc.).\n"
        "Do not add any other text."
    )

    for model_name in GEMINI_MODELS:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            import google.api_core.exceptions as gexc

            response = gemini_model.generate_content([
                {"mime_type": mime, "data": base64.b64encode(image_bytes).decode("utf-8")},
                prompt
            ])

            answer = response.text.strip().upper()
            logger.info(f"Gemini ({model_name}) mammogram check: {response.text.strip()!r}")

            if answer.startswith("YES"):
                return True, ""
            else:
                reason = response.text.strip()
                if ":" in reason:
                    reason = reason.split(":", 1)[1].strip()
                return False, reason

        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "resource" in err_str or "429" in err_str or "exhausted" in err_str:
                logger.warning(f"Gemini ({model_name}) quota exceeded, trying next model...")
                continue  # try next model
            else:
                logger.warning(f"Gemini ({model_name}) error ({e}); allowing image through as fallback.")
                return True, ""  # non-quota error → fail open

    # All models exhausted quota → fail open so the app still works
    logger.warning("All Gemini models hit quota limits; skipping validation and allowing image through.")
    return True, ""


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Load tabular model (Pipeline: StandardScaler + SVM) ──────────────────────
try:
    model = pickle.load(open('model.pkl', 'rb'))
    logger.info("Tabular model loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load model.pkl: {e}")
    model = None

# ── Load image classification model ──────────────────────────────────────────
try:
    logger.info("Loading image analysis model...")
    image_model = model_utils.load_cancer_model()
    if image_model is not None:
        logger.info("Image analysis model loaded.")
    else:
        logger.warning("Image analysis model returned None (weights may be missing).")
except Exception as e:
    logger.warning(f"Failed to load image model: {e}")
    image_model = None

# Feature names for the Wisconsin dataset (9 features)
ORIGINAL_FEATURES = [
    'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
    'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

# Feature names for the comprehensive 30-feature dataset
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]


def generate_insights(values, feature_names, prediction):
    """Generate medical insights based on the input values."""
    insights = []
    try:
        values_array = np.array(values)
        p75 = np.percentile(values_array, 75)
        p25 = np.percentile(values_array, 25)

        high_values = [name.replace('_', ' ').title() for name, v in zip(feature_names, values) if v > p75]
        low_values  = [name.replace('_', ' ').title() for name, v in zip(feature_names, values) if v < p25]

        if high_values:
            insights.append(f"Elevated values detected in: {', '.join(high_values[:3])}")
        if low_values:
            insights.append(f"Lower values observed in: {', '.join(low_values[:3])}")

        if prediction == 'Malignant':
            insights.append("Multiple cellular abnormalities detected requiring immediate medical attention")
            insights.append("Recommend immediate consultation with oncologist")
        else:
            insights.append("Cellular characteristics within normal ranges")
            insights.append("Continue regular screening as recommended by physician")
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        insights.append("Could not generate detailed insights due to data format.")

    return insights


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template(
                'index.html',
                error="Prediction model is not loaded. Please run train_model.py first."
            )

        input_features = []
        feature_names_used = []

        # Try original Wisconsin features first
        for feature in ORIGINAL_FEATURES:
            if feature in request.form:
                val = request.form[feature]
                if val:
                    input_features.append(float(val))
                    feature_names_used.append(feature)

        # Fall back to comprehensive 30-feature set if original not provided
        if not input_features:
            for feature in FEATURE_NAMES:
                if feature in request.form:
                    val = request.form[feature]
                    if val:
                        input_features.append(float(val))
                        feature_names_used.append(feature)

        if not input_features:
            return render_template('index.html', error="No valid input features provided")

        # The model is a Pipeline (StandardScaler + SVM), so we feed raw values
        features_array = np.array(input_features).reshape(1, -1)

        # Predict class and probability
        output = model.predict(features_array)
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_array)[0].tolist()

        # Interpret result (Wisconsin: 2 = Benign, 4 = Malignant)
        pred_val = output[0]
        if pred_val == 4 or pred_val == 1:
            result = "Malignant"
            risk_level = "High Risk"
            color_class = "danger"
        else:
            result = "Benign"
            risk_level = "Low Risk"
            color_class = "success"

        # Confidence score from model probability
        if prediction_proba:
            confidence = round(max(prediction_proba) * 100, 2)
        else:
            # Fallback: use decision function distance if available
            confidence = 0.0
            logger.warning("predict_proba not available; confidence set to 0.")

        insights = generate_insights(input_features, feature_names_used, result)

        return render_template(
            'index.html',
            prediction_text=f'Prediction: {result}',
            risk_level=risk_level,
            color_class=color_class,
            confidence=confidence,
            feature_names=feature_names_used,
            feature_values=input_features,
            prediction_probs=prediction_proba,
            insights=insights,
            show_results=True
        )

    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        return render_template('index.html', error=f"Error in prediction: {str(e)}")


@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Endpoint for image-based breast cancer detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # ── Gemini pre-filter: verify this is actually a mammogram ────────────
        file.seek(0)
        image_bytes = file.read()
        
        # If model is loaded, verify it's a mammogram first
        # If model is NOT loaded, we will use Gemini for prediction anyway, so verification is implicit
        is_valid, rejection_reason = is_mammogram_image(image_bytes)
        if not is_valid:
            msg = (
                f"The uploaded image does not appear to be a mammogram or breast scan. "
                f"{rejection_reason + ' ' if rejection_reason else ''}"
                f"Please upload a valid mammogram, breast ultrasound, or histology image."
            )
            return jsonify({'error': msg}), 422

        # ── Local Model Prediction ───────────────────────────────────────────
        # Use local model ONLY if it has valid weights loaded
        if image_model is not None and getattr(image_model, 'weights_loaded', False):
            processed_img = model_utils.preprocess_image(file)
            if processed_img is None:
                return jsonify({'error': 'Failed to process image. Please upload a valid image file.'}), 400

            predictions = image_model.predict(processed_img)[0]
            logger.info(f"Raw image predictions: {predictions}")

            result, confidence, density, full_results = model_utils.get_prediction_result(predictions)

            if result == "Error":
                return jsonify({
                    'error': 'Model produced invalid output. The weights may be incompatible.'
                }), 500

            if result == "NotMammogram":
                # Fallback to pure error if model rejects it (though check above should catch it)
                return jsonify({
                    'error': 'The uploaded image was rejected by the local model confidence check.'
                }), 422

            return jsonify({
                'status': 'success',
                'prediction': result,
                'confidence': float(confidence * 100),
                'density': density,
                'details': full_results
            })

        # ── Fallback: Gemini Prediction (if local model missing or unweighted) ──
        else:
            logger.info("Local model missing or unweighted; utilizing Gemini for prediction.")
            return predict_with_gemini_fallback(image_bytes)

    except Exception as e:
        logger.error(f"Error in /predict-image: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during image analysis.'}), 500

def predict_with_gemini_fallback(image_bytes):
    """
    Uses Gemini to perform the full diagnosis when the local DenseNet model is missing.
    Returns a Flask JSON response matching the frontend's expected format.
    """
    try:
        # Detect MIME type
        mime = "image/png" if image_bytes[:8] == b'\x89PNG\r\n\x1a\n' else "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = (
            "Analyze this mammogram image for breast cancer detection. "
            "Provide a diagnosis (Benign or Malignant), a confidence score (0-100), "
            "and an estimated breast density category (1-4). "
            "Also provide a short medical explanation for your assessment. "
            "Output STRICT JSON format: "
            '{"diagnosis": "Benign/Malignant", "confidence": <number>, "density": "1/2/3/4", "explanation": "..."}'
        )
        
        # Try models in chain
        response_text = None
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                # Configure generation config for JSON response if possible (gemini-1.5+ supports it)
                generation_config = {"response_mime_type": "application/json"}
                
                resp = model.generate_content(
                    [{"mime_type": mime, "data": b64}, prompt],
                    generation_config=generation_config
                )
                response_text = resp.text
                break
            except Exception as e:
                logger.warning(f"Gemini prediction failed on {model_name}: {e}")
                continue
        
        if not response_text:
            return jsonify({'error': 'Image analysis service represents unavailable (Quota exceeded). Please try again later.'}), 503

        # Parse JSON
        import json
        try:
            data = json.loads(response_text)
            diagnosis = data.get("diagnosis", "Unknown")
            confidence = float(data.get("confidence", 0))
            density = str(data.get("density", "Unknown"))
            explanation = data.get("explanation", "AI-generated assessment.")
            
            return jsonify({
                'status': 'success',
                'prediction': diagnosis,
                'confidence': confidence,
                'density': density,
                'details': {
                    'Diagnosis': diagnosis,
                    'Explanation': explanation,
                    'Source': 'AI Analysis (Fallback)'
                }
            })
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini JSON: {response_text}")
            return jsonify({'error': 'AI analysis returned invalid format.'}), 500

    except Exception as e:
        logger.error(f"Gemini fallback error: {e}")
        return jsonify({'error': 'AI analysis service failed.'}), 500


if __name__ == "__main__":
    app.run(debug=True)