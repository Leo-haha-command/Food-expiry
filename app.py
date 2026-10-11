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

def convert_to_scanned_doc(image_path):
    """
    Converts a photo with uneven lighting/reflections into a clean 
    'scanned document' style binary image before OCR processing.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Shadow division trick to cancel lighting variations on curved plastic
    blur = cv2.GaussianBlur(denoised, (35, 35), 0)
    divided = cv2.divide(denoised, blur, scale=255)

    # Otsu thresholding for sharp black-and-white contrast
    _, scanned_doc = cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return scanned_doc

def group_ocr_lines(ocr_results, y_tolerance=18):
    """
    Groups OCR bounding boxes on the same horizontal line into single sentences.
    Prevents 'EXP' and '10 01 2027' from being split into separate tokens.
    """
    if not ocr_results:
        return []

    # Sort bounding boxes top-to-bottom by Y center
    sorted_results = sorted(ocr_results, key=lambda res: (res[0][0][1] + res[0][2][1]) / 2.0)
    lines = []

    for box, text, conf in sorted_results:
        y_center = (box[0][1] + box[2][1]) / 2.0
        placed = False

        for line in lines:
            if abs(line['y_center'] - y_center) < y_tolerance:
                line['text'] += " " + text
                line['confs'].append(conf)
                line['y_center'] = (line['y_center'] + y_center) / 2.0
                placed = True
                break

        if not placed:
            lines.append({'text': text, 'confs': [conf], 'y_center': y_center})

    return [(l['text'], sum(l['confs']) / len(l['confs'])) for l in lines]

def extract_features(text_string):
    length = len(text_string)
    digit_count = sum(c.isdigit() for c in text_string)
    separator_count = (text_string.count('/') + text_string.count('-') + 
                       text_string.count('.') + text_string.count(' '))
    has_keywords = 1 if re.search(r'(exp|bb|best|use|prod)', text_string, re.I) else 0
    digit_ratio = digit_count / max(length, 1)
    return np.array([[length, digit_count, separator_count, has_keywords, digit_ratio]])

def format_date_to_wording(date_string):
    try:
        # Extract purely the date component from full text line (e.g. "10 01 2027")
        match = re.search(r'(\b\d{2}[\s/.-]\d{2}[\s/.-]\d{2,4}\b|\b\d{4}[\s/.-]\d{2}[\s/.-]\d{2}\b)', date_string)
        target_str = match.group(0) if match else date_string

        dt = parse(target_str, fuzzy=True)
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

def process_image_expiry(grouped_ocr_results):
    date_candidates = []

    # Matches dates formatted with spaces, slashes, dashes, or dots
    date_regex = re.compile(r'(\b\d{2}[\s/.-]\d{2}[\s/.-]\d{2,4}\b|\b\d{4}[\s/.-]\d{2}[\s/.-]\d{2}\b)')

    for (text, confidence) in grouped_ocr_results:
        if confidence < 0.20:
            continue

        features = extract_features(text)
        is_expiry = classifier.predict(features)[0] if classifier else 0
        regex_match = date_regex.search(text)

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

    # 1. Convert photo into a clean scanned document format
    scanned_image = convert_to_scanned_doc(filepath)

    # 2. Extract OCR lines from scanned image (or original fallback)
    raw_results = reader.readtext(scanned_image if scanned_image is not None else filepath)

    # 3. Group OCR fragments into horizontal lines
    grouped_lines = group_ocr_lines(raw_results)

    full_text = " | ".join([text for (text, _) in grouped_lines])
    detected_dates = process_image_expiry(grouped_lines)

    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        "full_text": full_text,
        "expiry_dates": detected_dates
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
