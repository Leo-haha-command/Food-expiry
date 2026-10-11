import os
import re
import cv2
from datetime import datetime
from dateutil.parser import parse
import joblib
import numpy as np
import easyocr
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

try:
    classifier = joblib.load("model.pkl")
    print("Trained AI model loaded successfully.")
except FileNotFoundError:
    classifier = None
    print("Warning: model.pkl not found! Run train.py first.")

def preprocess_image(image_path):
    """Applies glare correction, dot-matrix joining, and thresholding."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_dots = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    binary = cv2.adaptiveThreshold(
        connected_dots, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return binary

def extract_features(text_string):
    length = len(text_string)
    digit_count = sum(c.isdigit() for c in text_string)
    slash_dash_count = text_string.count('/') + text_string.count('-') + text_string.count('.')
    has_keywords = 1 if re.search(r'(exp|bb|best|use)', text_string, re.I) else 0
    digit_ratio = digit_count / max(length, 1)
    return np.array([[length, digit_count, slash_dash_count, has_keywords, digit_ratio]])

def format_date_to_wording(date_string):
    try:
        dt = parse(date_string, fuzzy=True)
        wording = dt.strftime("%B %d, %Y")
        
        today = datetime.now()
        days_left = (dt - today).days

        if days_left < 0:
            status = f"Expired {abs(days_left)} days ago"
        elif days_left == 0:
            status = "Expires today!"
        else:
            status = f"Valid for {days_left} more days"

        return f"{wording} ({status})"
    except Exception:
        return date_string

def process_image_expiry(ocr_results):
    date_candidates = []

    for (_, text, confidence) in ocr_results:
        if confidence < 0.25:
            continue

        features = extract_features(text)
        is_expiry = classifier.predict(features)[0] if classifier else 0
        regex_match = re.search(r'(\b\d{2}[/.-]\d{2}[/.-]\d{2,4}\b|\b\d{4}[/.-]\d{2}[/.-]\d{2}\b)', text)

        if is_expiry or regex_match:
            formatted_wording = format_date_to_wording(text)
            date_candidates.append({
                "raw_text": text,
                "wording": formatted_wording,
                "confidence": round(confidence * 100, 2),
                "ai_verified": bool(is_expiry)
            })

    return date_candidates

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 1. Preprocess 3D/Curved surface image
    processed_img = preprocess_image(filepath)
    
    # 2. Run OCR on processed array (or fallback to original path if preprocessing fails)
    if processed_img is not None:
        results = reader.readtext(processed_img)
    else:
        results = reader.readtext(filepath)

    full_text = " ".join([text for (_, text, _) in results])
    detected_dates = process_image_expiry(results)

    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        "full_text": full_text,
        "expiry_dates": detected_dates
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
