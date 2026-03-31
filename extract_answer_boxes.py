import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "boxes"
DEBUG_DIR = OUTPUT_DIR / "_debug"

# Set True to wipe previous box output on each run
CLEAR_OUTPUT = True

# Numbering behavior
GLOBAL_NUMBERING = False
GLOBAL_START = 1
# Per-page start numbers (used when GLOBAL_NUMBERING = False)
PAGE_START_QUESTION = {
    "answer_sheet_pg_1": 1,
    "answer_sheet_pg_2": 1,
}

# --------- TUNING PARAMS ---------
ADAPTIVE_BLOCK = 31
ADAPTIVE_C = 12

H_LINE_SCALE = 35
V_LINE_SCALE = 35
ROW_LINE_THRESH_RATIO = 0.15
VERT_LINE_MIN_HEIGHT_PCT = 0.6

BAND_MIN_HEIGHT = 20
BAND_MAX_HEIGHT = 60
MIN_BAND_INK = 250
MIN_BAND_WIDTH_RATIO = 0.2
IGNORE_TOP_PCT = 0.12

PAD_X = 2
PAD_Y = 0
BAND_PAD_TOP = 3
BAND_PAD_BOTTOM = 3

# Deskew
DESKEW = True
DESKEW_MAX_ANGLE = 8.0


def _adaptive_bin(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK,
        ADAPTIVE_C,
    )


def _deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    h, w = gray.shape
    min_len = int(w * 0.35)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=20,
    )
    if lines is None:
        return img, 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # normalize to [-90, 90]
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 30:
            angles.append(angle)

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return img, 0.0

    # clamp to avoid wild rotations
    median_angle = max(-DESKEW_MAX_ANGLE, min(DESKEW_MAX_ANGLE, median_angle))

    # rotate by negative angle to deskew
    angle = -median_angle
    (h, w) = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, angle


def _detect_grid_lines(bw, img_w, img_h):
    h_len = max(30, img_w // H_LINE_SCALE)
    v_len = max(30, img_h // V_LINE_SCALE)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    return h_lines, v_lines


def _row_bands_from_lines(h_lines, ink, img_h, img_w):
    row_sum = h_lines.sum(axis=1) / 255.0
    thr = img_w * ROW_LINE_THRESH_RATIO
    peaks = [i for i, v in enumerate(row_sum) if v > thr]

    ranges = []
    for i in peaks:
        if not ranges or i > ranges[-1][1] + 1:
            ranges.append([i, i])
        else:
            ranges[-1][1] = i

    ys = [(r[0] + r[1]) // 2 for r in ranges]
    ys = [0] + ys + [img_h - 1]
    ys = sorted(set(ys))

    bands = []
    top_ignore = int(img_h * IGNORE_TOP_PCT)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        h = y1 - y0
        if h < BAND_MIN_HEIGHT or h > BAND_MAX_HEIGHT:
            continue
        if y0 < top_ignore:
            continue
        band_ink = cv2.countNonZero(ink[y0:y1, :])
        if band_ink < MIN_BAND_INK:
            continue
        bands.append((y0, y1))
    return bands


def _line_positions_in_band(v_lines, y0, y1):
    band = v_lines[y0:y1, :]
    col_sum = band.sum(axis=0) / 255.0
    thr = (y1 - y0) * VERT_LINE_MIN_HEIGHT_PCT
    cols = np.where(col_sum > thr)[0]

    ranges = []
    for c in cols:
        if not ranges or c > ranges[-1][1] + 1:
            ranges.append([c, c])
        else:
            ranges[-1][1] = c

    centers = [(r[0] + r[1]) // 2 for r in ranges]
    return centers


def _pick_reference_lines(v_lines, bands, img_h):
    line_sets = []
    for (y0, y1) in bands:
        lines = _line_positions_in_band(v_lines, y0, y1)
        if len(lines) >= 5:
            line_sets.append(lines)

    if line_sets:
        return max(line_sets, key=len)

    # Fallback: global detection with a looser threshold
    col_sum = v_lines.sum(axis=0) / 255.0
    thr = img_h * 0.4
    cols = np.where(col_sum > thr)[0]
    ranges = []
    for c in cols:
        if not ranges or c > ranges[-1][1] + 1:
            ranges.append([c, c])
        else:
            ranges[-1][1] = c
    return [(r[0] + r[1]) // 2 for r in ranges]


def _question_start_for_page(stem, num_cols, row_count, global_state):
    if GLOBAL_NUMBERING:
        start = global_state["current"]
        global_state["current"] += num_cols * row_count
        return start
    return PAGE_START_QUESTION.get(stem, 1)


def process_image(image_path, csv_rows, global_state):
    img = cv2.imread(str(image_path))
    if img is None:
        print("Skipped (could not read):", image_path)
        return

    if DESKEW:
        img, angle = _deskew_image(img)
        if abs(angle) > 0.01:
            print(f"Deskewed {image_path.name}: {angle:.2f} deg")

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = _adaptive_bin(gray)
    h_lines, v_lines = _detect_grid_lines(bw, img_w, img_h)
    grid = cv2.bitwise_or(h_lines, v_lines)
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    bands = _row_bands_from_lines(h_lines, ink, img_h, img_w)
    if not bands:
        print("No answer bands found:", image_path.name)
        return

    ref_lines = _pick_reference_lines(v_lines, bands, img_h)
    if len(ref_lines) < 2:
        print("Not enough vertical lines found:", image_path.name)
        return

    ref_lines = sorted(ref_lines)
    num_cols = len(ref_lines) - 1

    start_q = _question_start_for_page(image_path.stem, num_cols, len(bands), global_state)

    out_dir = OUTPUT_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = img.copy()

    for r_idx, (y0, y1) in enumerate(sorted(bands, key=lambda b: b[0])):
        row_lines = _line_positions_in_band(v_lines, y0, y1)
        lines = ref_lines
        if len(row_lines) == len(ref_lines):
            lines = sorted(row_lines)

        # Expand the band slightly so top/bottom strokes aren't clipped
        y0b = max(y0 - BAND_PAD_TOP, 0)
        y1b = min(y1 + BAND_PAD_BOTTOM, img_h - 1)

        for c_idx in range(len(lines) - 1):
            x0 = lines[c_idx]
            x1 = lines[c_idx + 1]

            x0p = max(x0 + PAD_X, 0)
            x1p = min(x1 - PAD_X, img_w - 1)
            y0p = max(y0b + PAD_Y, 0)
            y1p = min(y1b - PAD_Y, img_h - 1)
            if x1p <= x0p or y1p <= y0p:
                continue

            q_num = start_q + (r_idx * num_cols) + c_idx
            out_name = f"q{q_num:03d}_r{r_idx + 1:02d}_c{c_idx + 1:02d}.png"
            out_path = out_dir / out_name
            crop = img[y0p:y1p, x0p:x1p]
            ok = cv2.imwrite(str(out_path), crop)
            if not ok:
                print("Failed to write:", out_path)
                continue

            csv_rows.append(
                [image_path.name, r_idx + 1, c_idx + 1, q_num, str(out_path)]
            )

            cv2.rectangle(debug, (x0p, y0p), (x1p, y1p), (0, 255, 0), 1)

    dbg_dir = DEBUG_DIR / image_path.stem
    dbg_dir.mkdir(parents=True, exist_ok=True)
    if DESKEW:
        cv2.imwrite(str(dbg_dir / "deskew.png"), img)
    cv2.imwrite(str(dbg_dir / "cells.png"), debug)

    print(f"Done: {image_path.name} -> {out_dir}")


def main():
    if CLEAR_OUTPUT and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in (".png", ".jpg", ".jpeg"):
        images.extend(sorted(INPUT_DIR.glob(f"*{ext}")))

    if not images:
        print("No images found in:", INPUT_DIR)
        return

    csv_rows = [["page", "row", "col", "question", "path"]]
    global_state = {"current": GLOBAL_START}

    for img_path in images:
        process_image(img_path, csv_rows, global_state)

    csv_path = OUTPUT_DIR / "labels.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print("Saved labels:", csv_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = OUTPUT_DIR / f"labels_{ts}.csv"
        with alt_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print("labels.csv was locked; saved labels to:", alt_path)


if __name__ == "__main__":
    main()
