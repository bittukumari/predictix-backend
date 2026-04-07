from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import random
import time

# Try to import TensorFlow, but don't crash if it's not installed yet
try:
    import numpy as np
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed. Running in MOCK MODE.")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# 1. ATTEMPT TO LOAD REAL MODEL
# ==========================================
MODEL_PATH = 'mobilenetv2_finetuned.h5'
LABEL_MAP_PATH = 'label_map.json'

model = None
LABELS = []
USE_MOCK_MODE = False

if TF_AVAILABLE and os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    print("Loading Machine Learning Model... This might take a few seconds.")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            LABELS = json.load(f)['labels']
        print("✅ Real Model and Labels loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading real model: {e}. Falling back to MOCK MODE.")
        USE_MOCK_MODE = True
else:
    print("⚠️ Model files not found. Starting in MOCK MODE.")
    USE_MOCK_MODE = True
    # These are the actual classes from the SIPaKMeD dataset you are using!
    LABELS = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

# ==========================================
# 2. IMAGE PREPROCESSING (For Real Mode)
# ==========================================
def preprocess_image_for_model(filepath):
    img_raw = tf.io.read_file(filepath)
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)

# ==========================================
# 3. THE PREDICTION API ROUTE
# ==========================================
@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if USE_MOCK_MODE:
                # --- MOCK MODE ---
                time.sleep(1.5) # Fake a delay so the loading spinner shows on frontend
                predicted_class_name = random.choice(LABELS)
                confidence_score = round(random.uniform(75.0, 99.5), 1)
                
                probabilities = {label: round(random.uniform(0.01, 0.1), 3) for label in LABELS}
                probabilities[predicted_class_name] = confidence_score / 100.0

            else:
                # --- REAL MODE ---
                processed_img = preprocess_image_for_model(filepath)
                predictions = model.predict(processed_img)[0]
                
                max_index = np.argmax(predictions)
                confidence_score = float(predictions[max_index]) * 100
                predicted_class_name = LABELS[max_index]
                probabilities = {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}

            # Clean up the temp image
            os.remove(filepath)

            # Send result to Next.js frontend
            return jsonify({
                "riskLevel": predicted_class_name,
                "confidence": confidence_score,
                "all_probabilities": probabilities,
                "is_mock": USE_MOCK_MODE
            })

        except Exception as e:
            print("Error during prediction:", e)
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)