import json
import os
import re
import cv2
import easyocr
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image_path):
    """
    Preprocesses 3D, curved, or dot-matrix printed product images 
    using OpenCV before feeding them to EasyOCR.
    """
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2. Reduce glare on curved/shiny surfaces (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 3. Connect sparse dot-matrix characters using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_dots = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 4. Adaptive thresholding to isolate text under uneven 3D shadows/lighting
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
    return [length, digit_count, slash_dash_count, has_keywords, digit_ratio]

DATASET_DIR = "dataset"
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

X_train = []
y_train = []

# Baseline synthetic samples to ensure minimum training data
synthetic_samples = [
    ("EXP 12/08/2027", 1),
    ("BEST BEFORE 05-2026", 1),
    ("BB 10/26", 1),
    ("NET WT 500g", 0),
    ("INGREDIENTS: WATER", 0),
    ("LOT 88492021", 0)
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

        # Step A: Preprocess the 3D/Curved Image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            print(f"Warning: Failed to load image {image_path}. Skipping.")
            continue

        # Step B: Pass the processed image array into EasyOCR
        ocr_results = reader.readtext(processed_img)

        for (_, extracted_text, confidence) in ocr_results:
            if confidence > 0.3:
                X_train.append(extract_features(extracted_text))
                y_train.append(label_info["is_expiry"])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "model.pkl")
print("AI Model successfully trained and saved to model.pkl")

