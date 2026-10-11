import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings


import gc
import json
import re
import cv2
import easyocr
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize EasyOCR Reader
print("Initializing EasyOCR for training...")
reader = easyocr.Reader(['en'], gpu=False)


def preprocess_image(image_path, target_max_dim=800):
    """
    Downscales images to a lighter max dimension (800px) during training 
    to prevent RAM exhaustion and process termination in memory-constrained environments.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    max_dim = max(h, w)
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def group_ocr_lines(ocr_results, y_tolerance=18):
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
    length = len(text_string)
    digit_count = sum(c.isdigit() for c in text_string)
    separator_count = (text_string.count('/') + text_string.count('-') + 
                       text_string.count('.') + text_string.count(' '))
    has_keywords = 1 if re.search(r'(exp|bb|best|use|prod)', text_string, re.I) else 0
    digit_ratio = digit_count / max(length, 1)
    return [length, digit_count, separator_count, has_keywords, digit_ratio]


def main():
    DATASET_DIR = "dataset"
    ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

    X_train = []
    y_train = []

    # 1. Synthetic baseline samples
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

    # 2. Load dataset annotations
    image_label_map = {}
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                for img_name, info in annotations.items():
                    image_label_map[img_name] = info.get("is_expiry", 0)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse '{ANNOTATIONS_FILE}'.")

    # 3. Process images with explicit memory cleanup
    if image_label_map:
        total = len(image_label_map)
        print(f"Processing {total} tagged images from '{DATASET_DIR}/'...")

        for idx, (image_name, is_expiry_label) in enumerate(image_label_map.items(), start=1):
            image_path = os.path.join(DATASET_DIR, image_name)
            if not os.path.exists(image_path):
                continue

            print(f"[{idx}/{total}] Processing {image_name}...")
            processed_img = preprocess_image(image_path)
            if processed_img is None:
                continue

            try:
                raw_results = reader.readtext(processed_img) or []
                grouped_lines = group_ocr_lines(raw_results)

                for (text_line, confidence) in grouped_lines:
                    if confidence > 0.3:
                        has_date_pattern = bool(re.search(
                            r'(\b\d{1,2}[\s/.-]\d{1,2}[\s/.-]\d{2,4}\b|\b(?:exp|bb|mfg|use|prod)[\s:]*\d+|\b\d{6,8}\b)', 
                            text_line, re.I
                        ))
                        line_label = 1 if (is_expiry_label == 1 and has_date_pattern) else 0

                        X_train.append(extract_features(text_line))
                        y_train.append(line_label)
            except Exception as e:
                print(f"Skipping {image_name} due to OCR error: {e}")

            # Explicit memory cleanup after each image
            del processed_img
            gc.collect()

    # 4. Fit and export Random Forest Classifier
    print(f"Training Random Forest model on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "model.pkl")
    print("AI Model successfully retrained and saved to model.pkl")


if __name__ == '__main__':
    main()