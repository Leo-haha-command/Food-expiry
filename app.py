import os
import re
import cv2
import json
import uuid
import shutil
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import joblib
import easyocr
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Active Learning Folders
DATASET_FOLDER = 'dataset'
REVIEW_FOLDER = os.path.join(DATASET_FOLDER, 'review_queue')
ANNOTATIONS_FILE = os.path.join(DATASET_FOLDER, 'annotations.json')
os.makedirs(REVIEW_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Load trained model if present
try:
    classifier = joblib.load("model.pkl")
    print("Trained AI model loaded successfully.")
except FileNotFoundError:
    classifier = None
    print("Warning: model.pkl not found! Run train.py first.")


@app.route('/favicon.ico')
@app.route('/apple-touch-icon.png')
@app.route('/apple-touch-icon-precomposed.png')
def ignore_icon_requests():
    """Suppresses browser 404 icon requests in terminal logs."""
    return '', 204


def preprocess_for_ocr(image_path, target_max_dim=1600):
    """
    Dual-pass preprocessing:
    1. Grayscale with CLAHE contrast enhancement (best for standard fonts)
    2. Morphological closing + Otsu thresholding (fallback for dot-matrix inkjet)
    Includes resolution capping to avoid Out-Of-Memory crashes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Memory safety: Scale images to a safe pixel dimension
    max_dim = max(h, w)
    if max_dim < 800:
        scale = 2.0  # Upscale low-res stamps
    elif max_dim > target_max_dim:
        scale = target_max_dim / max_dim  # Downscale large photos
    else:
        scale = 1.0

    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Pass 1: Enhanced grayscale for standard printed text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Pass 2: Morphological closing for dot-matrix inkjet text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    closed = cv2.morphologyEx(enhanced_gray, cv2.MORPH_CLOSE, kernel)
    _, binary_img = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return enhanced_gray, binary_img


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
    if not grouped_ocr_results:
        return date_candidates

    for (text, confidence) in grouped_ocr_results:
        if not text or confidence < 0.20:
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
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Generate preprocessed grayscale & binary images
        enhanced_gray, binary_img = preprocess_for_ocr(filepath)

        if enhanced_gray is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Failed to read or decode image"}), 400

        # Pass 1: High-accuracy grayscale OCR
        raw_results = reader.readtext(enhanced_gray) or []
        grouped_lines = group_ocr_lines(raw_results)
        detected_dates = process_image_expiry(grouped_lines)

        # Pass 2: Fall back to dot-matrix binary OCR if Pass 1 yields no dates
        if not detected_dates:
            raw_results_fallback = reader.readtext(binary_img) or []
            grouped_lines_fallback = group_ocr_lines(raw_results_fallback)
            detected_dates_fallback = process_image_expiry(grouped_lines_fallback)
            if detected_dates_fallback:
                grouped_lines = grouped_lines_fallback
                detected_dates = detected_dates_fallback
            elif not grouped_lines and grouped_lines_fallback:
                grouped_lines = grouped_lines_fallback

        full_text = " | ".join([text for (text, _) in grouped_lines if text]) if grouped_lines else "No text detected on image"

        # Save preprocessed image temporarily in review queue
        image_id = f"{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}.jpg"
        review_filepath = os.path.join(REVIEW_FOLDER, image_id)
        cv2.imwrite(review_filepath, enhanced_gray)

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "image_id": image_id,
            "full_text": full_text,
            "expiry_dates": detected_dates or []
        })

    except Exception as e:
        print("Server Error during prediction:", str(e))
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Moves image to dataset/ and updates dataset/annotations.json directly.
    """
    try:
        data = request.get_json() or {}
        image_id = data.get('image_id')
        is_expiry = data.get('is_expiry')

        if not image_id or is_expiry not in [0, 1]:
            return jsonify({"error": "Invalid payload. 'image_id' and 'is_expiry' (0 or 1) required."}), 400

        source_path = os.path.join(REVIEW_FOLDER, image_id)
        target_path = os.path.join(DATASET_FOLDER, image_id)

        # 1. Move image into the main dataset folder
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)

        # 2. Read existing annotations.json (or create dictionary if missing)
        annotations = {}
        if os.path.exists(ANNOTATIONS_FILE):
            try:
                with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = {}

        # 3. Add or update entry for this image
        annotations[image_id] = {"is_expiry": is_expiry}

        # 4. Save updated dictionary back to annotations.json
        with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=4)

        return jsonify({"status": "success", "message": f"Updated dataset/annotations.json for {image_id}"})

    except Exception as e:
        return jsonify({"error": f"Failed to record feedback: {str(e)}"}), 500


@app.route('/admin/retrain', methods=['POST'])
def retrain():
    """
    Executes train.py and reloads the updated model into active memory.
    """
    global classifier
    try:
        exit_code = os.system("python train.py")
        if exit_code == 0 and os.path.exists("model.pkl"):
            classifier = joblib.load("model.pkl")
            return jsonify({"status": "success", "message": "Model retrained and loaded into memory."})
        else:
            return jsonify({"error": "Training script execution failed."}), 500

    except Exception as e:
        return jsonify({"error": f"Retraining error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)