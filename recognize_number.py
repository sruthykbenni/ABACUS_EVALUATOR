# recognize_number.py
import cv2
import numpy as np
from skimage.feature import hog
from segment_digits import segment_digits
from image_classifier import deskew

def preprocess_single_digit(digit, scaler):
    digit = deskew(digit)

    density = np.count_nonzero(digit) / digit.size
    if density < 0.12:
        digit = cv2.dilate(digit, np.ones((2,2), np.uint8), 1)

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    resized = cv2.resize(
        digit, (int(w*scale), int(h*scale)),
        interpolation=cv2.INTER_NEAREST
    )

    padded = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - resized.shape[0]) // 2
    x0 = (28 - resized.shape[1]) // 2
    padded[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized
    padded = padded / 255.0

    features = hog(
        padded,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    return scaler.transform([features])

def recognize_number(image, model, scaler):
    digits = segment_digits(image)

    number = ""
    confidences = []

    for d in digits:
        features = preprocess_single_digit(d, scaler)
        probs = model.predict_proba(features)[0]
        digit = int(np.argmax(probs))
        conf = float(np.max(probs))

        confidences.append(conf)

        if conf < 0.70 or (digit == 1 and conf < 0.80):
            number += "?"
        else:
            number += str(digit)

    avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
    return number, avg_conf
