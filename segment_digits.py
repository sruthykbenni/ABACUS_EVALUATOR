# segment_digits.py
import cv2
import numpy as np

def segment_digits(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert for white paper
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # OTSU threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Fallback for very light pencil
    if np.count_nonzero(thresh) < 0.05 * thresh.size:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 15 or w < 8:
            continue
        digits.append(thresh[y:y+h, x:x+w])

    return digits
