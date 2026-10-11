import json
import os
import re
import cv2
import easyocr
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image_path):
    """
    Preprocesses 3D, curved, or dot-matrix printed product images 
    matching the app.py scanning pipeline.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def group_ocr_lines(ocr_results, y_tolerance=18):
    """
    Groups bounding boxes into horizontal lines matching app.py.
    Ensures training features evaluate full context (e.g. 'EXP : 10 01 2027').
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

def extract_features(text_string):
    """Identical 5-feature extraction vector as app.py."""
    length = len(text_string)
    digit_count = sum(c.isdigit() for c in text_string)
    separator_count = (text_string.count('/') + text_string.count('-') + 
                       text_string.count('.') + text_string.count(' '))
    has_keywords = 1 if re.search(r'(exp|bb|best|use|prod)', text_string, re.I) else 0
    digit_ratio = digit_count / max(length, 1)
    return [length, digit_count, separator_count, has_keywords, digit_ratio]

DATASET_DIR = "dataset"
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

X_train = []
y_train = []

# Expanded synthetic samples covering space-separated, dot-matrix, and concatenated formats
synthetic_samples = [
    ("EXP 12/08/2027", 1),
    ("EXP : 10 01 2027", 1),
    ("PROD : 10 01 2026", 1),
    ("BEST BEFORE 05-2026", 1),
    ("BB 10/26", 1),
    ("EXP 040917", 1),
    ("10 JAN 2027", 1),
    ("USE BY 15.08.2027", 1),
    ("NET WT 500g", 0),
    ("INGREDIENTS: WATER", 0),
    ("LOT 88492021", 0),
    ("NUTRITION FACTS", 0),
    ("248 kcal", 0)
]

for text, label in synthetic_samples:
    X_train.append(extract_features(text))
    y_train.append(label)

if os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, 'r') as f:
        dataset = json.load(f)

    print(f"Processing {len(dataset)} training images from '{DATASET_DIR}/'...")

    for image_name, label_info in dataset.items():
        image_path = os.path.join(DATASET_DIR, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            continue

        # Step A: Preprocess image using unified dot-matrix pipeline
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            print(f"Warning: Failed to load image {image_path}. Skipping.")
            continue

        # Step B: Run EasyOCR & group horizontal lines
        raw_results = reader.readtext(processed_img)
        grouped_lines = group_ocr_lines(raw_results)

        for (text_line, confidence) in grouped_lines:
            if confidence > 0.3:
                X_train.append(extract_features(text_line))
                y_train.append(label_info["is_expiry"])

# Train Random Forest Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "model.pkl")
print("AI Model successfully trained and saved to model.pkl")
