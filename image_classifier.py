# image_classifier.py
import cv2
import numpy as np
from skimage.feature import hog

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST)

def preprocess_image(image, scaler):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    digit = deskew(digit)

    # Adaptive dilation
    density = np.count_nonzero(digit) / digit.size
    if density < 0.12:
        digit = cv2.dilate(digit, np.ones((2,2), np.uint8), 1)

    # Aspect-ratio preserving resize
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
