import csv
import shutil
from dataclasses import dataclass
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
ROW_PAGE_START_QUESTION = {
    "answer_sheet_pg_1": 1,
    "answer_sheet_pg_2": 1,
}
COL_PAGE_START_QUESTION = {
    "paper_pg_1": 1,
    "paper_pg_2": 121,
}

# Shared preprocessing
ADAPTIVE_BLOCK = 31
ADAPTIVE_C = 12
DESKEW = True
DESKEW_MAX_ANGLE = 8.0

# Row-layout tuning
ROW_H_LINE_SCALE = 35
ROW_V_LINE_SCALE = 35
ROW_LINE_THRESH_RATIO = 0.15
ROW_VERT_LINE_MIN_HEIGHT_PCT = 0.6
ROW_BAND_MIN_HEIGHT = 20
ROW_BAND_MAX_HEIGHT = 60
ROW_MIN_BAND_INK = 250
ROW_IGNORE_TOP_PCT = 0.12
ROW_PAD_X = 2
ROW_PAD_Y = 0
ROW_BAND_PAD_TOP = 3
ROW_BAND_PAD_BOTTOM = 3

# Column-layout tuning
COL_H_LINE_SCALE = 25
COL_V_LINE_SCALE = 30
COL_ROW_LINE_THRESH_RATIO = 0.03
COL_BAND_MIN_HEIGHT = 10
COL_BAND_MAX_HEIGHT = 70
COL_MIN_BAND_INK = 50
COL_IGNORE_TOP_PCT = 0.05
COL_PAD_X = 2
COL_PAD_Y = 0
COL_BAND_PAD_TOP = 2
COL_BAND_PAD_BOTTOM = 2
EXPECTED_ROWS = 30
EXPECTED_ANSWER_COLS = 4
MIN_ANSWER_X_FRAC = 0.20
WIDEST_BANDS_KEEP = 4

# Auto-detection tuning
ROW_GAP_CV_MAX = 0.18
ROW_GAP_CLOSE_TOL_PCT = 0.25
ROW_GAP_CLOSE_TOL_MIN = 3
ROW_MIN_UNIFORM_GAP_RATIO = 0.75
ROW_MIN_MATCHED_ROW_RATIO = 0.75
MIN_COL_LAYOUT_ANSWER_BANDS = 2
MIN_COL_LAYOUT_ROWS = 10


@dataclass
class RowLayoutCandidate:
    bands: list
    ref_lines: list
    gap_cv: float
    gap_close_ratio: float
    matched_rows: int
    matched_row_ratio: float
    v_lines: np.ndarray | None
    valid: bool

    @property
    def num_cols(self):
        return max(len(self.ref_lines) - 1, 0)


@dataclass
class ColLayoutCandidate:
    row_bands: list
    answer_bands: list
    valid: bool


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

    median_angle = max(-DESKEW_MAX_ANGLE, min(DESKEW_MAX_ANGLE, median_angle))
    angle = -median_angle
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, angle


def _detect_grid_lines(bw, img_w, img_h, h_scale, v_scale):
    h_len = max(20, img_w // h_scale)
    v_len = max(20, img_h // v_scale)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    return h_lines, v_lines


def _row_bands_from_lines(
    h_lines,
    ink,
    img_h,
    img_w,
    line_thresh_ratio,
    band_min_height,
    band_max_height,
    min_band_ink,
    ignore_top_pct,
):
    row_sum = h_lines.sum(axis=1) / 255.0
    thr = img_w * line_thresh_ratio
    peaks = [i for i, value in enumerate(row_sum) if value > thr]

    ranges = []
    for idx in peaks:
        if not ranges or idx > ranges[-1][1] + 1:
            ranges.append([idx, idx])
        else:
            ranges[-1][1] = idx

    ys = [(start + end) // 2 for start, end in ranges]
    ys = [0] + ys + [img_h - 1]
    ys = sorted(set(ys))

    bands = []
    top_ignore = int(img_h * ignore_top_pct)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        band_h = y1 - y0
        if band_h < band_min_height or band_h > band_max_height:
            continue
        if y0 < top_ignore:
            continue
        band_ink = cv2.countNonZero(ink[y0:y1, :])
        if band_ink < min_band_ink:
            continue
        bands.append((y0, y1))
    return bands


def _line_positions_in_band(v_lines, y0, y1, min_height_pct):
    band = v_lines[y0:y1, :]
    col_sum = band.sum(axis=0) / 255.0
    thr = (y1 - y0) * min_height_pct
    cols = np.where(col_sum > thr)[0]

    ranges = []
    for c in cols:
        if not ranges or c > ranges[-1][1] + 1:
            ranges.append([c, c])
        else:
            ranges[-1][1] = c

    return [(start + end) // 2 for start, end in ranges]


def _pick_reference_lines(v_lines, bands, img_h, min_height_pct):
    line_sets = []
    for y0, y1 in bands:
        lines = _line_positions_in_band(v_lines, y0, y1, min_height_pct)
        if len(lines) >= 5:
            line_sets.append(lines)

    if line_sets:
        return max(line_sets, key=len)

    col_sum = v_lines.sum(axis=0) / 255.0
    thr = img_h * 0.4
    cols = np.where(col_sum > thr)[0]
    ranges = []
    for c in cols:
        if not ranges or c > ranges[-1][1] + 1:
            ranges.append([c, c])
        else:
            ranges[-1][1] = c
    return [(start + end) // 2 for start, end in ranges]


def _col_bands_from_lines(v_lines, img_h):
    col_sum = v_lines.sum(axis=0) / 255.0
    thr = img_h * COL_ROW_LINE_THRESH_RATIO
    peaks = [i for i, value in enumerate(col_sum) if value > thr]

    ranges = []
    for idx in peaks:
        if not ranges or idx > ranges[-1][1] + 1:
            ranges.append([idx, idx])
        else:
            ranges[-1][1] = idx

    xs = [(start + end) // 2 for start, end in ranges]
    xs = [0] + xs
    return sorted(set(xs))


def _column_spans(col_lines, img_w):
    xs = sorted(set(col_lines + [img_w - 1]))
    spans = []
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 - x0 < COL_BAND_MIN_HEIGHT:
            continue
        spans.append((x0, x1))
    return spans


def _select_answer_bands(col_bands, img_w):
    if not col_bands:
        return []

    widths = np.array([x1 - x0 for x0, x1 in col_bands], dtype=np.int32)
    min_x = img_w * MIN_ANSWER_X_FRAC

    if len(widths) >= 2:
        sorted_widths = np.sort(widths)
        gaps = np.diff(sorted_widths)
        gap_idx = int(np.argmax(gaps))
        narrow = sorted_widths[: gap_idx + 1]
        wide = sorted_widths[gap_idx + 1 :]
        if narrow.size and wide.size and wide.mean() >= narrow.mean() * 1.4:
            wide_thr = (sorted_widths[gap_idx] + sorted_widths[gap_idx + 1]) / 2.0
            wide_mask = widths >= wide_thr
            answers = []
            run_start = None
            for idx, is_wide in enumerate(wide_mask):
                if is_wide:
                    if run_start is None:
                        run_start = idx
                    continue
                if run_start is not None:
                    answers.append(col_bands[idx - 1])
                    run_start = None
            if run_start is not None:
                answers.append(col_bands[len(col_bands) - 1])

            answers = [band for band in answers if band[1] > min_x]
            if answers:
                return answers

    candidates = [band for band in col_bands if band[1] > min_x]
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda band: ((band[1] - band[0]), band[1]),
        reverse=True,
    )
    return sorted(ranked[:WIDEST_BANDS_KEEP], key=lambda band: band[0])


def _normalize_row_bands(bands):
    bands = sorted(bands, key=lambda band: band[0])
    if EXPECTED_ROWS <= 0 or len(bands) <= EXPECTED_ROWS:
        return bands

    best_window = bands[:EXPECTED_ROWS]
    best_score = None
    for start in range(len(bands) - EXPECTED_ROWS + 1):
        window = bands[start : start + EXPECTED_ROWS]
        heights = [y1 - y0 for y0, y1 in window]
        gaps = [window[i + 1][0] - window[i][0] for i in range(len(window) - 1)]
        score = float(np.var(heights) + np.var(gaps))
        if best_score is None or score < best_score:
            best_score = score
            best_window = window
    return best_window


def _normalize_answer_bands(answer_bands):
    answer_bands = sorted(answer_bands, key=lambda band: band[0])
    if EXPECTED_ANSWER_COLS <= 0 or len(answer_bands) <= EXPECTED_ANSWER_COLS:
        return answer_bands

    ranked = sorted(
        answer_bands,
        key=lambda band: ((band[1] - band[0]), band[1]),
        reverse=True,
    )
    trimmed = ranked[:EXPECTED_ANSWER_COLS]
    return sorted(trimmed, key=lambda band: band[0])


def _build_row_candidate(bw, img_h, img_w):
    h_lines, v_lines = _detect_grid_lines(
        bw,
        img_w,
        img_h,
        ROW_H_LINE_SCALE,
        ROW_V_LINE_SCALE,
    )
    grid = cv2.bitwise_or(h_lines, v_lines)
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    bands = _row_bands_from_lines(
        h_lines,
        ink,
        img_h,
        img_w,
        ROW_LINE_THRESH_RATIO,
        ROW_BAND_MIN_HEIGHT,
        ROW_BAND_MAX_HEIGHT,
        ROW_MIN_BAND_INK,
        ROW_IGNORE_TOP_PCT,
    )
    if not bands:
        return RowLayoutCandidate([], [], float("inf"), 0.0, 0, 0.0, None, False)

    ref_lines = sorted(
        _pick_reference_lines(v_lines, bands, img_h, ROW_VERT_LINE_MIN_HEIGHT_PCT)
    )
    if len(ref_lines) < 2:
        return RowLayoutCandidate(
            bands,
            ref_lines,
            float("inf"),
            0.0,
            0,
            0.0,
            v_lines,
            False,
        )

    gaps = np.diff(ref_lines)
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) and np.mean(gaps) else float("inf")
    if len(gaps):
        median_gap = float(np.median(gaps))
        tol = max(ROW_GAP_CLOSE_TOL_MIN, median_gap * ROW_GAP_CLOSE_TOL_PCT)
        gap_close_ratio = float(np.mean(np.abs(gaps - median_gap) <= tol))
    else:
        gap_close_ratio = 0.0
    row_counts = [
        len(_line_positions_in_band(v_lines, y0, y1, ROW_VERT_LINE_MIN_HEIGHT_PCT))
        for y0, y1 in bands
    ]
    matched_rows = sum(1 for count in row_counts if count == len(ref_lines))
    matched_row_ratio = float(matched_rows / len(bands)) if bands else 0.0

    return RowLayoutCandidate(
        bands,
        ref_lines,
        gap_cv,
        gap_close_ratio,
        matched_rows,
        matched_row_ratio,
        v_lines,
        True,
    )


def _build_col_candidate(bw, img_h, img_w):
    h_lines, v_lines = _detect_grid_lines(
        bw,
        img_w,
        img_h,
        COL_H_LINE_SCALE,
        COL_V_LINE_SCALE,
    )
    grid = cv2.bitwise_or(h_lines, v_lines)
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(grid))

    row_bands = _normalize_row_bands(
        _row_bands_from_lines(
            h_lines,
            ink,
            img_h,
            img_w,
            COL_ROW_LINE_THRESH_RATIO,
            COL_BAND_MIN_HEIGHT,
            COL_BAND_MAX_HEIGHT,
            COL_MIN_BAND_INK,
            COL_IGNORE_TOP_PCT,
        )
    )
    col_lines = _col_bands_from_lines(v_lines, img_h)
    col_spans = _column_spans(col_lines, img_w)
    answer_bands = _normalize_answer_bands(_select_answer_bands(col_spans, img_w))

    valid = bool(row_bands and answer_bands)
    return ColLayoutCandidate(row_bands, answer_bands, valid)


def _choose_layout(row_candidate, col_candidate):
    row_confident = (
        row_candidate.valid
        and row_candidate.gap_close_ratio >= ROW_MIN_UNIFORM_GAP_RATIO
        and row_candidate.matched_row_ratio >= ROW_MIN_MATCHED_ROW_RATIO
    )
    col_confident = (
        col_candidate.valid
        and len(col_candidate.answer_bands) >= MIN_COL_LAYOUT_ANSWER_BANDS
        and len(col_candidate.row_bands) >= MIN_COL_LAYOUT_ROWS
    )

    if row_confident and not col_confident:
        return "row"

    if col_confident and not row_confident:
        return "col"

    if (
        col_confident
        and row_candidate.gap_cv > ROW_GAP_CV_MAX
    ):
        return "col"

    if row_candidate.valid and (
        row_candidate.gap_cv <= ROW_GAP_CV_MAX
        or row_candidate.gap_close_ratio >= ROW_MIN_UNIFORM_GAP_RATIO
    ):
        return "row"

    if col_candidate.valid and len(col_candidate.row_bands) < MIN_COL_LAYOUT_ROWS:
        return "row" if row_candidate.valid else "col"

    if col_confident:
        return "col"

    if row_candidate.valid:
        return "row"

    if col_candidate.valid:
        return "col"

    return None


def _question_start_for_page(stem, layout, num_cols, row_count, global_state):
    if GLOBAL_NUMBERING:
        start = global_state["current"]
        global_state["current"] += num_cols * row_count
        return start

    if layout == "row":
        return ROW_PAGE_START_QUESTION.get(stem, 1)
    return COL_PAGE_START_QUESTION.get(stem, 1)


def _crop_row_layout(img, image_path, candidate, csv_rows, global_state):
    img_h, img_w = img.shape[:2]
    start_q = _question_start_for_page(
        image_path.stem,
        "row",
        candidate.num_cols,
        len(candidate.bands),
        global_state,
    )

    out_dir = OUTPUT_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = img.copy()

    for r_idx, (y0, y1) in enumerate(sorted(candidate.bands, key=lambda band: band[0])):
        row_lines = _line_positions_in_band(
            candidate.v_lines,
            y0,
            y1,
            ROW_VERT_LINE_MIN_HEIGHT_PCT,
        )
        lines = candidate.ref_lines
        if len(row_lines) == len(candidate.ref_lines):
            lines = sorted(row_lines)

        y0b = max(y0 - ROW_BAND_PAD_TOP, 0)
        y1b = min(y1 + ROW_BAND_PAD_BOTTOM, img_h - 1)

        for c_idx in range(len(lines) - 1):
            x0 = lines[c_idx]
            x1 = lines[c_idx + 1]

            x0p = max(x0 + ROW_PAD_X, 0)
            x1p = min(x1 - ROW_PAD_X, img_w - 1)
            y0p = max(y0b + ROW_PAD_Y, 0)
            y1p = min(y1b - ROW_PAD_Y, img_h - 1)
            if x1p <= x0p or y1p <= y0p:
                continue

            q_num = start_q + (r_idx * candidate.num_cols) + c_idx
            out_name = f"q{q_num:03d}_r{r_idx + 1:02d}_c{c_idx + 1:02d}.png"
            out_path = out_dir / out_name
            crop = img[y0p:y1p, x0p:x1p]
            if not cv2.imwrite(str(out_path), crop):
                print("Failed to write:", out_path)
                continue

            csv_rows.append(
                ["row", image_path.name, r_idx + 1, c_idx + 1, q_num, str(out_path)]
            )
            cv2.rectangle(debug, (x0p, y0p), (x1p, y1p), (0, 255, 0), 1)

    return debug


def _crop_col_layout(img, image_path, candidate, csv_rows, global_state):
    img_h, img_w = img.shape[:2]
    row_bands = sorted(candidate.row_bands, key=lambda band: band[0])
    answer_bands = sorted(candidate.answer_bands, key=lambda band: band[0])
    row_count = len(row_bands)
    start_q = _question_start_for_page(
        image_path.stem,
        "col",
        len(answer_bands),
        row_count,
        global_state,
    )

    out_dir = OUTPUT_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = img.copy()

    for y0, y1 in row_bands:
        cv2.line(debug, (0, y0), (img_w - 1, y0), (0, 255, 255), 1)
        cv2.line(debug, (0, y1), (img_w - 1, y1), (0, 255, 255), 1)
    for x0, x1 in answer_bands:
        cv2.line(debug, (x0, 0), (x0, img_h - 1), (255, 200, 0), 1)
        cv2.line(debug, (x1, 0), (x1, img_h - 1), (255, 200, 0), 1)

    for c_idx, (x0, x1) in enumerate(answer_bands):
        for r_idx, (y0, y1) in enumerate(row_bands):
            y0b = max(y0 - COL_BAND_PAD_TOP, 0)
            y1b = min(y1 + COL_BAND_PAD_BOTTOM, img_h - 1)
            x0p = max(x0 + COL_PAD_X, 0)
            x1p = min(x1 - COL_PAD_X, img_w - 1)
            y0p = max(y0b + COL_PAD_Y, 0)
            y1p = min(y1b - COL_PAD_Y, img_h - 1)
            if x1p <= x0p or y1p <= y0p:
                continue

            q_num = start_q + (c_idx * row_count) + r_idx
            out_name = f"q{q_num:03d}_r{r_idx + 1:02d}_c{c_idx + 1:02d}.png"
            out_path = out_dir / out_name
            crop = img[y0p:y1p, x0p:x1p]
            if not cv2.imwrite(str(out_path), crop):
                print("Failed to write:", out_path)
                continue

            csv_rows.append(
                ["col", image_path.name, r_idx + 1, c_idx + 1, q_num, str(out_path)]
            )
            cv2.rectangle(debug, (x0p, y0p), (x1p, y1p), (0, 255, 0), 1)

    return debug


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

    row_candidate = _build_row_candidate(bw, img_h, img_w)
    col_candidate = _build_col_candidate(bw, img_h, img_w)
    layout = _choose_layout(row_candidate, col_candidate)

    if layout is None:
        print("No supported layout found:", image_path.name)
        return

    dbg_dir = DEBUG_DIR / image_path.stem
    dbg_dir.mkdir(parents=True, exist_ok=True)
    if DESKEW:
        cv2.imwrite(str(dbg_dir / "deskew.png"), img)

    if layout == "row":
        debug = _crop_row_layout(img, image_path, row_candidate, csv_rows, global_state)
        cv2.imwrite(str(dbg_dir / "cells_row.png"), debug)
        print(
            f"Done: {image_path.name} -> layout=row "
            f"({len(row_candidate.bands)} rows x {row_candidate.num_cols} cols, "
            f"gap_cv={row_candidate.gap_cv:.3f}, "
            f"uniform_gap_ratio={row_candidate.gap_close_ratio:.3f})"
        )
        return

    debug = _crop_col_layout(img, image_path, col_candidate, csv_rows, global_state)
    cv2.imwrite(str(dbg_dir / "cells_col.png"), debug)
    print(
        f"Done: {image_path.name} -> layout=col "
        f"({len(col_candidate.row_bands)} rows x {len(col_candidate.answer_bands)} cols, "
        f"answer_bands={len(col_candidate.answer_bands)}, "
        f"row_gap_cv={row_candidate.gap_cv:.3f}, "
        f"row_uniform_gap_ratio={row_candidate.gap_close_ratio:.3f})"
    )


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    csv_rows = [["layout", "page", "row", "col", "question", "path"]]
    global_state = {"current": GLOBAL_START}

    for img_path in images:
        process_image(img_path, csv_rows, global_state)

    csv_path = OUTPUT_DIR / "labels.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(csv_rows)
        print("Saved labels:", csv_path)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = OUTPUT_DIR / f"labels_{ts}.csv"
        with alt_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(csv_rows)
        print("labels.csv was locked; saved labels to:", alt_path)


if __name__ == "__main__":
    main()
