import os
import re
import cv2
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import joblib
import easyocr
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Load trained model if present
try:
    classifier = joblib.load("model.pkl")
    print("Trained AI model loaded successfully.")
except FileNotFoundError:
    classifier = None
    print("Warning: model.pkl not found! Run train.py first.")


def convert_to_scanned_doc(image_path):
    """
    Applies Gaussian blurring and shadow division to bridge dot-matrix 
    inkjet characters on curved surfaces before binarization.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Light Gaussian blur bridges gaps between matrix dots
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Equalize contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 3. Morphological closing to join adjacent dots horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 4. Otsu Thresholding
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def group_ocr_lines(ocr_results, y_tolerance=18):
    """
    Groups OCR bounding boxes on the same horizontal Y-axis into unified text strings.
    Prevents splitting key labels like 'EXP:' from dates like '10 01 2027'.
    """
    if not ocr_results:
        return []

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


def convert_to_status_wording(date_str):
    """Converts extracted date string into natural wording and remaining days."""
    try:
        dt = parse(date_str, fuzzy=True)
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
        return date_str


def extract_and_format_date(text_line):
    """
    Extracts date patterns from raw OCR strings across all packaging formats:
    1. Text months: '10 JAN 2027', 'EXP 04-SEP-26', 'OCT 2027'
    2. Separated numeric: '10/01/2027', '04-09-17', '10.01.27', '10 01 2027'
    3. Concatenated numeric: '040917', '20270110'
    """
    clean_line = text_line.strip()

    # Pattern 1: Dates with named months
    text_month_match = re.search(
        r'(\b\d{1,2}[\s/.-]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/.-]?\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/.-]?\d{2,4}\b)',
        clean_line, re.I
    )
    if text_month_match:
        target_str = text_month_match.group(0)
        return convert_to_status_wording(target_str)

    # Pattern 2: Standard separated dates with '/', '-', '.', or spaces
    separated_match = re.search(
        r'(\b\d{1,2}[\s/.-]\d{1,2}[\s/.-]\d{2,4}\b|\b\d{4}[\s/.-]\d{1,2}[\s/.-]\d{1,2}\b)',
        clean_line
    )
    if separated_match:
        target_str = separated_match.group(0)
        return convert_to_status_wording(target_str)

    # Pattern 3: Concatenated digits following EXP/MFG/BB keywords or standalone
    concat_match = re.search(r'(?:exp|bb|mfg|use\s*by)?[\s:]*(\b\d{6}\b|\b\d{8}\b)', clean_line, re.I)
    if concat_match:
        digits = concat_match.group(1)
        if len(digits) == 6:
            target_str = f"{digits[:2]}/{digits[2:4]}/20{digits[4:]}"
        elif len(digits) == 8:
            if digits.startswith("20"):  # YYYYMMDD format
                target_str = f"{digits[:4]}/{digits[4:6]}/{digits[6:]}"
            else:  # DDMMYYYY format
                target_str = f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
        return convert_to_status_wording(target_str)

    return clean_line


def process_image_expiry(grouped_ocr_results):
    date_candidates = []

    for (text, confidence) in grouped_ocr_results:
        if confidence < 0.20:
            continue

        formatted_wording = extract_and_format_date(text)

        if formatted_wording != text or re.search(r'(exp|bb|best|use)', text, re.I):
            date_candidates.append({
                "raw_text": text,
                "wording": formatted_wording,
                "confidence": round(confidence * 100, 2)
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

    scanned_image = convert_to_scanned_doc(filepath)
    raw_results = reader.readtext(scanned_image if scanned_image is not None else filepath)

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
    app.run(host='0.0.0.0', port=5000, debug=True)
