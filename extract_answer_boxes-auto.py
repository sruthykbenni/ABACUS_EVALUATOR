import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import cv2
import fitz
import numpy as np

from crop_config import *

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


@dataclass
class InputPage:
    display_name: str
    output_stem: str
    numbering_stem: str
    image: np.ndarray


def _adaptive_bin(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK,
        ADAPTIVE_C,
    )


def _normalize_line_angle(angle):
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle


def _weighted_median(values, weights):
    if not values:
        return None

    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None

    cumulative = 0.0
    midpoint = total_weight / 2.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= midpoint:
            return float(value)
    return float(pairs[-1][0])


def _order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _quad_side_lengths(points):
    tl, tr, br, bl = points
    widths = (
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - bl)),
    )
    heights = (
        float(np.linalg.norm(bl - tl)),
        float(np.linalg.norm(br - tr)),
    )
    return widths, heights


def _quad_output_size(points):
    widths, heights = _quad_side_lengths(points)
    out_w = int(round(max(widths)))
    out_h = int(round(max(heights)))
    return out_w, out_h


def _quad_max_angle_cos(points):
    ordered = _order_quad_points(points)

    def _corner_cos(prev_pt, corner_pt, next_pt):
        v1 = prev_pt - corner_pt
        v2 = next_pt - corner_pt
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom <= 1e-6:
            return 1.0
        return abs(float(np.dot(v1, v2) / denom))

    return max(
        _corner_cos(ordered[(idx - 1) % 4], ordered[idx], ordered[(idx + 1) % 4])
        for idx in range(4)
    )


def _valid_page_quad(points, area, img_area):
    if points is None or area <= 0 or img_area <= 0:
        return False

    ordered = _order_quad_points(points)
    out_w, out_h = _quad_output_size(ordered)
    if min(out_w, out_h) < PERSPECTIVE_MIN_OUTPUT_SIDE:
        return False

    aspect_ratio = max(out_w, out_h) / max(min(out_w, out_h), 1)
    if aspect_ratio > PERSPECTIVE_MAX_ASPECT_RATIO:
        return False

    if (area / img_area) < PERSPECTIVE_MIN_AREA_RATIO:
        return False

    if _quad_max_angle_cos(ordered) > PERSPECTIVE_MAX_ANGLE_COS:
        return False

    return True


def _quad_edge_support(edges, points):
    ordered = _order_quad_points(points)
    if edges is None or edges.size == 0:
        return 0.0, []

    thickness = max(
        3,
        int(
            round(
                max(
                    PERSPECTIVE_EDGE_SUPPORT_THICKNESS,
                    min(edges.shape[:2]) * 0.003,
                )
            )
        ),
    )
    scores = []
    for idx in range(4):
        p0 = tuple(np.round(ordered[idx]).astype(np.int32))
        p1 = tuple(np.round(ordered[(idx + 1) % 4]).astype(np.int32))
        line_mask = np.zeros_like(edges)
        cv2.line(line_mask, p0, p1, 255, thickness=thickness)
        line_pixels = cv2.countNonZero(line_mask)
        if line_pixels <= 0:
            scores.append(0.0)
            continue
        overlap = cv2.countNonZero(cv2.bitwise_and(line_mask, edges))
        scores.append(float(overlap / line_pixels))

    return min(scores) if scores else 0.0, scores


def _detect_page_quad_from_color_border(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue border mask (tweak H range if needed)
    lower = np.array([90, 40, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # pick largest contour and approximate quad
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 0:
        return None

    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return None

    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return _order_quad_points(approx.reshape(4, 2))

    # fallback to minAreaRect box
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return _order_quad_points(box)


def _detect_page_quad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = gray.shape
    img_area = float(orig_h * orig_w)

    scale = 1.0
    max_dim = max(orig_h, orig_w)
    if max_dim > PERSPECTIVE_DETECT_MAX_DIM:
        scale = PERSPECTIVE_DETECT_MAX_DIM / float(max_dim)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # Try color-border based detection first (works well when page has colored frame)
    try:
        color_quad = _detect_page_quad_from_color_border(img)
        if color_quad is not None:
            # validate color_quad using edge support (strict threshold)
            min_support, _ = _quad_edge_support(edges, color_quad)
            if _valid_page_quad(color_quad, float(gray.shape[0] * gray.shape[1]), float(orig_h * orig_w)) and min_support >= 0.50:
                return color_quad
    except Exception:
        pass

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    scaled_area = float(gray.shape[0] * gray.shape[1])
    approx_candidates = []
    fallback_candidates = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, PERSPECTIVE_APPROX_EPS_RATIO * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = _order_quad_points(approx.reshape(4, 2)) / scale
            scaled_to_orig_area = area / (scale * scale)
            min_support, _ = _quad_edge_support(edges, approx.reshape(4, 2))
            if (
                _valid_page_quad(points, scaled_to_orig_area, img_area)
                and min_support >= PERSPECTIVE_MIN_EDGE_SUPPORT
            ):
                approx_candidates.append((scaled_to_orig_area, points))
            continue

        if (area / scaled_area) < PERSPECTIVE_FALLBACK_MIN_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        points = _order_quad_points(box) / scale
        scaled_to_orig_area = area / (scale * scale)
        min_support, _ = _quad_edge_support(edges, box)
        if (
            _valid_page_quad(points, scaled_to_orig_area, img_area)
            and min_support >= PERSPECTIVE_MIN_EDGE_SUPPORT
        ):
            fallback_candidates.append((scaled_to_orig_area, points))

    candidates = approx_candidates if approx_candidates else fallback_candidates
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _should_apply_perspective_warp(img, quad):
    return _quad_distortion_ratio(img.shape[:2], quad) >= PERSPECTIVE_MIN_DISTORTION_RATIO


def _quad_distortion_ratio(img_shape, quad):
    ordered = _order_quad_points(quad)
    (top_w, bottom_w), (left_h, right_h) = _quad_side_lengths(ordered)
    tl, tr, br, bl = ordered

    img_h, img_w = img_shape[:2]
    width_diff = abs(top_w - bottom_w) / max(top_w, bottom_w, 1.0)
    height_diff = abs(left_h - right_h) / max(left_h, right_h, 1.0)
    axis_drift = max(
        abs(tr[1] - tl[1]) / max(img_h, 1),
        abs(br[1] - bl[1]) / max(img_h, 1),
        abs(bl[0] - tl[0]) / max(img_w, 1),
        abs(br[0] - tr[0]) / max(img_w, 1),
    )
    return float(max(width_diff, height_diff, axis_drift))


def _warp_from_quad(img, quad, expand_ratio):
    ordered = _order_quad_points(quad)
    center = ordered.mean(axis=0)
    ordered = center + (ordered - center) * expand_ratio
    out_w, out_h = _quad_output_size(ordered)
    if min(out_w, out_h) < PERSPECTIVE_MIN_OUTPUT_SIDE:
        return None

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _warp_page_from_quad(img, quad):
    return _warp_from_quad(img, quad, PERSPECTIVE_WARP_EXPAND_RATIO)


def _perspective_correct_image(img):
    quad = _detect_page_quad(img)
    if quad is None:
        return img, None
    if not _should_apply_perspective_warp(img, quad):
        return img, None

    warped = _warp_page_from_quad(img, quad)
    if warped is None:
        return img, None

    return warped, quad


def _estimate_skew_from_mask(mask, min_len, threshold, max_line_gap, angle_limit):
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    angles = []
    lengths = []
    for x1, y1, x2, y2 in lines[:, 0]:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue

        angle = _normalize_line_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) > angle_limit:
            continue

        angles.append(float(angle))
        lengths.append(length)

    if len(angles) < DESKEW_MIN_LINE_COUNT:
        return None
    if sum(lengths) < min_len * DESKEW_MIN_LINE_COUNT:
        return None

    return _weighted_median(angles, lengths)


def _rotate_image_expand(img, angle):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    w = gray.shape[1]
    bw = _adaptive_bin(gray)

    grid_kernel_len = max(25, w // ROW_H_LINE_SCALE)
    grid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_kernel_len, 1))
    horizontal_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, grid_kernel)
    horizontal_mask = cv2.dilate(
        horizontal_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1,
    )

    median_angle = _estimate_skew_from_mask(
        horizontal_mask,
        min_len=max(30, int(w * 0.20)),
        threshold=50,
        max_line_gap=25,
        angle_limit=DESKEW_GRID_ANGLE_LIMIT,
    )
    if median_angle is None:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        median_angle = _estimate_skew_from_mask(
            edges,
            min_len=max(30, int(w * 0.35)),
            threshold=80,
            max_line_gap=20,
            angle_limit=DESKEW_FALLBACK_ANGLE_LIMIT,
        )
    if median_angle is None:
        return img, 0.0

    if abs(median_angle) < 0.1:
        return img, 0.0

    median_angle = max(-DESKEW_MAX_ANGLE, min(DESKEW_MAX_ANGLE, median_angle))
    angle = -median_angle
    return _rotate_image_expand(img, angle), angle


def _preprocess_page_image(img):
    work_img = img.copy()
    perspective_quad = None
    rectified_img = None

    if PERSPECTIVE_CORRECTION:
        rectified_img, perspective_quad = _perspective_correct_image(work_img)
        if perspective_quad is not None:
            work_img = rectified_img

    return work_img, {
        "perspective_quad": perspective_quad,
        "rectified_image": rectified_img,
    }


def _detect_grid_roi(bw, img_w, img_h, h_scale, v_scale):
    if not GRID_ROI_ENABLE:
        return None

    h_lines, v_lines = _detect_grid_lines(bw, img_w, img_h, h_scale, v_scale)
    h_support = cv2.dilate(
        h_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    v_support = cv2.dilate(
        v_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )
    grid_support = cv2.bitwise_or(h_support, v_support)
    grid_support = cv2.dilate(
        grid_support,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (GRID_ROI_COMPONENT_DILATE, GRID_ROI_COMPONENT_DILATE),
        ),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        grid_support,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    img_area = float(img_w * img_h)
    component_boxes = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(w * h)
        if box_area <= 0:
            continue
        if (box_area / img_area) < GRID_ROI_COMPONENT_MIN_BOX_RATIO:
            continue
        component_boxes.append((x, y, x + w - 1, y + h - 1))

    used_component_boxes = bool(component_boxes)
    if used_component_boxes:
        x0 = min(box[0] for box in component_boxes)
        y0 = min(box[1] for box in component_boxes)
        x1 = max(box[2] for box in component_boxes)
        y1 = max(box[3] for box in component_boxes)
    else:
        intersections = cv2.bitwise_and(h_support, v_support)
        ys, xs = np.where(intersections > 0)
        if len(xs) < GRID_ROI_MIN_INTERSECTION_PIXELS:
            return None

        x0 = int(np.percentile(xs, GRID_ROI_LOW_PCT))
        x1 = int(np.percentile(xs, GRID_ROI_HIGH_PCT))
        y0 = int(np.percentile(ys, GRID_ROI_LOW_PCT))
        y1 = int(np.percentile(ys, GRID_ROI_HIGH_PCT))
        if x1 <= x0 or y1 <= y0:
            return None

    if x1 <= x0 or y1 <= y0:
        return None

    roi_h_lines = h_lines[:, x0 : x1 + 1]
    row_sum = roi_h_lines.sum(axis=1) / 255.0
    roi_line_width = max(x1 - x0 + 1, 1)
    strong_rows = np.where(row_sum > (roi_line_width * GRID_ROI_STRONG_ROW_RATIO))[0]
    if len(strong_rows):
        strong_y0 = int(strong_rows[0])
        top_segment = row_sum[y0:strong_y0] if strong_y0 > y0 else np.asarray([])
        top_secondary_ratio = (
            float(top_segment.max() / roi_line_width) if top_segment.size else 0.0
        )
        if (
            (strong_y0 - y0) >= GRID_ROI_STRONG_TRIM_MIN
            and top_secondary_ratio >= GRID_ROI_SECONDARY_ROW_RATIO
        ):
            y0 = strong_y0
        if y1 <= y0:
            return None

    roi_w = x1 - x0 + 1
    roi_h = y1 - y0 + 1
    if roi_w < (img_w * GRID_ROI_MIN_WIDTH_RATIO):
        return None
    if roi_h < (img_h * GRID_ROI_MIN_HEIGHT_RATIO):
        return None

    if used_component_boxes:
        pad_x = max(
            GRID_ROI_COMPONENT_PAD_X_MIN,
            int(round(roi_w * GRID_ROI_COMPONENT_PAD_X_RATIO)),
        )
        pad_y = max(
            GRID_ROI_COMPONENT_PAD_Y_MIN,
            int(round(roi_h * GRID_ROI_COMPONENT_PAD_Y_RATIO)),
        )
    else:
        pad_x = max(GRID_ROI_PAD_X_MIN, int(round(roi_w * GRID_ROI_PAD_X_RATIO)))
        pad_y = max(GRID_ROI_PAD_Y_MIN, int(round(roi_h * GRID_ROI_PAD_Y_RATIO)))
    x0 = max(0, x0 - pad_x)
    x1 = min(img_w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(img_h - 1, y1 + pad_y)

    return x0, y0, x1, y1


def _crop_to_grid_roi(img, gray, bw):
    img_h, img_w = img.shape[:2]
    roi = _detect_grid_roi(
        bw,
        img_w,
        img_h,
        GRID_H_LINE_SCALE,
        GRID_V_LINE_SCALE,
    )
    if roi is None:
        return img, gray, bw, None

    x0, y0, x1, y1 = roi
    return (
        img[y0 : y1 + 1, x0 : x1 + 1],
        gray[y0 : y1 + 1, x0 : x1 + 1],
        bw[y0 : y1 + 1, x0 : x1 + 1],
        roi,
    )


def _detect_grid_quad(bw, img_w, img_h):
    if not GRID_PERSPECTIVE_ENABLE:
        return None

    h_lines, v_lines = _detect_grid_lines(
        bw,
        img_w,
        img_h,
        GRID_H_LINE_SCALE,
        GRID_V_LINE_SCALE,
    )
    h_support = cv2.dilate(
        h_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=1,
    )
    v_support = cv2.dilate(
        v_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        iterations=1,
    )
    grid_support = cv2.bitwise_or(h_support, v_support)
    grid_support = cv2.dilate(
        grid_support,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (GRID_PERSPECTIVE_COMPONENT_DILATE, GRID_PERSPECTIVE_COMPONENT_DILATE),
        ),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        grid_support,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    img_area = float(img_w * img_h)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        if (area / img_area) < GRID_PERSPECTIVE_MIN_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            GRID_PERSPECTIVE_APPROX_EPS_RATIO * perimeter,
            True,
        )
        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = _order_quad_points(approx.reshape(4, 2))
        else:
            points = _order_quad_points(cv2.boxPoints(cv2.minAreaRect(contour)))

        if _valid_page_quad(points, area, img_area):
            candidates.append((area, points))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _perspective_align_grid_roi(img, gray, bw):
    img_h, img_w = img.shape[:2]
    quad = _detect_grid_quad(bw, img_w, img_h)
    if quad is None:
        return img, gray, bw, None

    if _quad_distortion_ratio(img.shape[:2], quad) < GRID_PERSPECTIVE_MIN_DISTORTION_RATIO:
        return img, gray, bw, None

    warped_img = _warp_from_quad(img, quad, GRID_PERSPECTIVE_WARP_EXPAND_RATIO)
    if warped_img is None:
        return img, gray, bw, None

    warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    warped_bw = _adaptive_bin(warped_gray)
    recropped_img, recropped_gray, recropped_bw, roi = _crop_to_grid_roi(
        warped_img,
        warped_gray,
        warped_bw,
    )
    if roi is not None:
        return recropped_img, recropped_gray, recropped_bw, quad

    return warped_img, warped_gray, warped_bw, quad


def _estimate_grid_alignment_angle(bw, img_w, img_h):
    h_lines, _ = _detect_grid_lines(
        bw,
        img_w,
        img_h,
        GRID_H_LINE_SCALE,
        GRID_V_LINE_SCALE,
    )
    ys, xs = np.where(h_lines > 0)
    if len(xs) < GRID_ROI_MIN_INTERSECTION_PIXELS:
        return 0.0

    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    rect_w, rect_h = rect[1]
    angle = float(rect[-1])
    if rect_w < rect_h:
        angle -= 90.0
    angle = _normalize_line_angle(angle)

    if abs(angle) < GRID_ALIGN_MIN_ANGLE or abs(angle) > GRID_ALIGN_MAX_ANGLE:
        return 0.0
    return angle


def _align_grid_roi(img, gray, bw):
    img_h, img_w = img.shape[:2]
    angle = _estimate_grid_alignment_angle(bw, img_w, img_h)
    if abs(angle) < GRID_ALIGN_MIN_ANGLE:
        return img, gray, bw, 0.0

    aligned_img = _rotate_image_expand(img, angle)
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.GaussianBlur(aligned_gray, (3, 3), 0)
    aligned_bw = _adaptive_bin(aligned_gray)
    recropped_img, recropped_gray, recropped_bw, roi = _crop_to_grid_roi(
        aligned_img,
        aligned_gray,
        aligned_bw,
    )
    if roi is not None:
        roi_w = roi[2] - roi[0] + 1
        roi_h = roi[3] - roi[1] + 1
        if (
            roi_w >= aligned_img.shape[1] * GRID_ALIGN_RECROP_MIN_WIDTH_RATIO
            and roi_h >= aligned_img.shape[0] * GRID_ALIGN_RECROP_MIN_HEIGHT_RATIO
        ):
            return recropped_img, recropped_gray, recropped_bw, angle

    return aligned_img, aligned_gray, aligned_bw, angle


def _prepare_page_for_layout(img, align_grid=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = _adaptive_bin(gray)
    pre_roi_img = img.copy()
    img, gray, bw, grid_roi = _crop_to_grid_roi(img, gray, bw)
    if not align_grid:
        return img, gray, bw, pre_roi_img, grid_roi, 0.0, None

    aligned_img, aligned_gray, aligned_bw, grid_align_angle = _align_grid_roi(img, gray, bw)
    warped_img, warped_gray, warped_bw, grid_perspective_quad = _perspective_align_grid_roi(
        img,
        gray,
        bw,
    )
    if grid_perspective_quad is None:
        return aligned_img, aligned_gray, aligned_bw, pre_roi_img, grid_roi, grid_align_angle, None

    aligned_score = _prepared_layout_score(aligned_bw)
    warped_score = _prepared_layout_score(warped_bw)
    if warped_score > aligned_score:
        return warped_img, warped_gray, warped_bw, pre_roi_img, grid_roi, 0.0, grid_perspective_quad

    return aligned_img, aligned_gray, aligned_bw, pre_roi_img, grid_roi, grid_align_angle, None


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
        if y0 == 0:
            continue
        band_ink = cv2.countNonZero(ink[y0:y1, :])
        if y1 <= top_ignore and band_ink < min_band_ink:
            continue
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


def _region_ink_density(ink, x0, x1, y0, y1):
    if ink is None or x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((x1 - x0) * (y1 - y0), 1)
    return float(cv2.countNonZero(ink[y0:y1, x0:x1]) / area)


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

    # Ignore isolated one-pixel peaks caused by handwriting or scan noise.
    ranges = [
        (start, end)
        for start, end in ranges
        if (end - start + 1) >= COL_MIN_LINE_RUN_WIDTH
    ]

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


def _estimate_answer_band_width(col_bands, img_w):
    min_x = img_w * MIN_ANSWER_X_FRAC
    widths = sorted(
        [(x1 - x0) for x0, x1 in col_bands if x1 > min_x],
        reverse=True,
    )
    if not widths:
        return None

    sample_size = max(EXPECTED_ANSWER_COLS * 2, EXPECTED_ANSWER_COLS)
    sample = widths[:sample_size]
    return float(np.median(np.asarray(sample, dtype=np.float32)))


def _merge_split_answer_spans(col_bands, img_w):
    if len(col_bands) < 2:
        return col_bands

    target_width = _estimate_answer_band_width(col_bands, img_w)
    if not target_width or target_width <= 0:
        return col_bands

    min_x = img_w * MIN_ANSWER_X_FRAC
    merged = []
    idx = 0
    while idx < len(col_bands):
        if idx + 1 < len(col_bands):
            left = col_bands[idx]
            right = col_bands[idx + 1]
            left_w = left[1] - left[0]
            right_w = right[1] - right[0]
            combined_w = right[1] - left[0]

            if (
                left[1] > min_x
                and right[1] > min_x
                and left_w <= target_width * COL_SPLIT_COMPONENT_MAX_RATIO
                and right_w <= target_width * COL_SPLIT_COMPONENT_MAX_RATIO
                and combined_w >= target_width * COL_SPLIT_COMBINED_MIN_RATIO
                and combined_w <= target_width * COL_SPLIT_COMBINED_MAX_RATIO
            ):
                merged.append((left[0], right[1]))
                idx += 2
                continue

        merged.append(col_bands[idx])
        idx += 1

    return merged


def _band_ink_density(ink, band):
    x0, x1 = band
    if ink is None or x1 <= x0:
        return float("inf")
    area = max((x1 - x0) * ink.shape[0], 1)
    return float(cv2.countNonZero(ink[:, x0:x1]) / area)


def _select_alternating_answer_bands(candidates, ink):
    if ink is None or len(candidates) < EXPECTED_ANSWER_COLS:
        return []

    widths = np.asarray([x1 - x0 for x0, x1 in candidates], dtype=np.float32)
    typical_width = float(np.median(widths)) if widths.size else 0.0
    filtered = [
        band
        for band in candidates
        if (band[1] - band[0]) >= typical_width * 0.6
    ]
    if len(filtered) < (EXPECTED_ANSWER_COLS * 2 - 1):
        return []

    best_window = []
    best_score = None
    for offset in (0, 1):
        seq = filtered[offset::2]
        if len(seq) < EXPECTED_ANSWER_COLS:
            continue

        for start in range(len(seq) - EXPECTED_ANSWER_COLS + 1):
            window = seq[start : start + EXPECTED_ANSWER_COLS]
            widths = np.asarray([x1 - x0 for x0, x1 in window], dtype=np.float32)
            densities = np.asarray(
                [_band_ink_density(ink, band) for band in window],
                dtype=np.float32,
            )
            score = (
                float(np.median(densities)),
                float(np.std(densities)),
                float(np.std(widths)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_window = window

    return sorted(best_window, key=lambda band: band[0])


def _select_answer_bands(col_bands, img_w, ink=None):
    if not col_bands:
        return []

    min_x = img_w * MIN_ANSWER_X_FRAC
    candidates = [band for band in col_bands if band[1] > min_x]
    if not candidates:
        return []

    alternating = _select_alternating_answer_bands(candidates, ink)
    if alternating:
        return alternating

    widths = np.array([x1 - x0 for x0, x1 in col_bands], dtype=np.int32)

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
            if len(answers) >= EXPECTED_ANSWER_COLS:
                return answers
            if answers:
                return answers

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


def _filter_row_answer_like_bands(bands, ink, x0, x1):
    if not bands:
        return []

    filtered = list(bands)
    heights = np.asarray([y1 - y0 for y0, y1 in filtered], dtype=np.float32)
    unique_heights = np.unique(heights)
    if len(unique_heights) > 1:
        sorted_heights = np.sort(unique_heights)
        gaps = np.diff(sorted_heights)
        split_idx = int(np.argmax(gaps))
        largest_gap = float(gaps[split_idx]) if len(gaps) else 0.0
        lower = sorted_heights[: split_idx + 1]
        upper = sorted_heights[split_idx + 1 :]
        if (
            largest_gap >= ROW_LOCAL_HEIGHT_GAP_MIN
            and lower.size
            and upper.size
            and float(np.median(upper))
            >= float(np.median(lower)) * ROW_LOCAL_HEIGHT_RATIO_MIN
        ):
            height_thr = float(
                (sorted_heights[split_idx] + sorted_heights[split_idx + 1]) / 2.0
            )
            filtered = [
                band for band in filtered if (band[1] - band[0]) >= height_thr
            ]

    return [
        (y0, y1)
        for y0, y1 in filtered
        if _region_ink_density(ink, x0, x1, y0, y1) >= ROW_LOCAL_MIN_INK_DENSITY
    ]


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


def _choose_best_col_row_bands(global_bands, local_bands):
    if not local_bands:
        return global_bands
    if not global_bands:
        return local_bands
    if EXPECTED_ROWS > 0:
        global_gap = abs(len(global_bands) - EXPECTED_ROWS)
        local_gap = abs(len(local_bands) - EXPECTED_ROWS)
        if local_gap < global_gap:
            return local_bands
        if global_gap < local_gap:
            return global_bands
    if len(local_bands) >= len(global_bands):
        return local_bands
    return global_bands


def _row_bands_from_answer_columns(h_lines, ink, answer_bands, img_h):
    local_sets = []
    for x0, x1 in sorted(answer_bands, key=lambda band: band[0]):
        local = _normalize_row_bands(
            _row_bands_from_lines(
                h_lines[:, x0:x1],
                ink[:, x0:x1],
                img_h,
                x1 - x0,
                COL_ROW_LINE_THRESH_RATIO,
                COL_BAND_MIN_HEIGHT,
                COL_BAND_MAX_HEIGHT,
                COL_MIN_BAND_INK,
                COL_IGNORE_TOP_PCT,
            )
        )
        if len(local) >= MIN_COL_LAYOUT_ROWS:
            local_sets.append(local)

    if not local_sets:
        return []

    counts = Counter(len(bands) for bands in local_sets)
    target_count = max(
        counts.items(),
        key=lambda item: (item[1], -abs(item[0] - EXPECTED_ROWS), item[0]),
    )[0]
    aligned_sets = [bands for bands in local_sets if len(bands) == target_count]

    consensus = []
    for idx in range(target_count):
        y0 = int(np.median([bands[idx][0] for bands in aligned_sets]))
        y1 = int(np.median([bands[idx][1] for bands in aligned_sets]))
        if y1 > y0:
            consensus.append((y0, y1))

    return consensus


def _row_bands_from_reference_columns(h_lines, ink, ref_lines, img_h):
    local_sets = []
    for x0, x1 in zip(ref_lines, ref_lines[1:]):
        if x1 - x0 < ROW_LOCAL_MIN_SPAN_WIDTH:
            continue
        local = _filter_row_answer_like_bands(
            _row_bands_from_lines(
                h_lines[:, x0:x1],
                ink[:, x0:x1],
                img_h,
                x1 - x0,
                ROW_LINE_THRESH_RATIO,
                ROW_BAND_MIN_HEIGHT,
                ROW_LOCAL_BAND_MAX_HEIGHT,
                ROW_LOCAL_MIN_BAND_INK,
                ROW_IGNORE_TOP_PCT,
            ),
            ink,
            x0,
            x1,
        )
        if local:
            local_sets.append(local)

    if len(local_sets) < ROW_LOCAL_MIN_SPANS:
        return []

    heights = [y1 - y0 for bands in local_sets for y0, y1 in bands]
    if not heights:
        return []
    cluster_tol = max(
        ROW_LOCAL_CLUSTER_TOL_MIN,
        int(np.median(np.asarray(heights, dtype=np.float32)) * ROW_LOCAL_CLUSTER_TOL_RATIO),
    )

    observations = []
    for span_idx, bands in enumerate(local_sets):
        for y0, y1 in bands:
            observations.append((span_idx, (y0 + y1) / 2.0, y0, y1))
    observations.sort(key=lambda item: item[1])

    clusters = []
    for observation in observations:
        center = observation[1]
        if not clusters or center - clusters[-1]["center"] > cluster_tol:
            clusters.append({"center": center, "items": [observation]})
            continue

        clusters[-1]["items"].append(observation)
        clusters[-1]["center"] = float(
            np.median([item[1] for item in clusters[-1]["items"]])
        )

    min_support = max(
        ROW_LOCAL_MIN_SUPPORT,
        int(np.ceil(len(local_sets) * ROW_LOCAL_MIN_SUPPORT_RATIO)),
    )
    consensus = []
    for cluster in clusters:
        support = len({item[0] for item in cluster["items"]})
        if support < min_support:
            continue
        y0 = int(np.median([item[2] for item in cluster["items"]]))
        y1 = int(np.median([item[3] for item in cluster["items"]]))
        if y1 > y0:
            consensus.append((y0, y1))

    return sorted(consensus, key=lambda band: band[0])


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

    seed_bands = _row_bands_from_lines(
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
    if not seed_bands:
        return RowLayoutCandidate([], [], float("inf"), 0.0, 0, 0.0, None, False)

    ref_lines = sorted(
        _pick_reference_lines(v_lines, seed_bands, img_h, ROW_VERT_LINE_MIN_HEIGHT_PCT)
    )
    if len(ref_lines) < 2:
        return RowLayoutCandidate(
            seed_bands,
            ref_lines,
            float("inf"),
            0.0,
            0,
            0.0,
            v_lines,
            False,
        )

    bands = _row_bands_from_reference_columns(h_lines, ink, ref_lines, img_h)
    if not bands:
        bands = seed_bands

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
    valid = (
        bool(bands)
        and len(bands) <= ROW_MAX_LAYOUT_ROWS
        and (len(ref_lines) - 1) >= ROW_MIN_LAYOUT_COLS
    )

    return RowLayoutCandidate(
        bands,
        ref_lines,
        gap_cv,
        gap_close_ratio,
        matched_rows,
        matched_row_ratio,
        v_lines,
        valid,
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
    col_spans = _merge_split_answer_spans(_column_spans(col_lines, img_w), img_w)
    answer_bands = _normalize_answer_bands(_select_answer_bands(col_spans, img_w, ink))
    local_row_bands = _row_bands_from_answer_columns(h_lines, ink, answer_bands, img_h)
    row_bands = _choose_best_col_row_bands(row_bands, local_row_bands)

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


def _prepared_layout_score(bw):
    img_h, img_w = bw.shape[:2]
    row_candidate = _build_row_candidate(bw, img_h, img_w)
    col_candidate = _build_col_candidate(bw, img_h, img_w)
    layout = _choose_layout(row_candidate, col_candidate)

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
    gap_score = (
        -row_candidate.gap_cv if np.isfinite(row_candidate.gap_cv) else float("-inf")
    )

    if layout == "col":
        row_gap = abs(len(col_candidate.row_bands) - EXPECTED_ROWS) if EXPECTED_ROWS > 0 else 0
        col_gap = (
            abs(len(col_candidate.answer_bands) - EXPECTED_ANSWER_COLS)
            if EXPECTED_ANSWER_COLS > 0
            else 0
        )
        return (
            3 if col_confident else 2,
            -row_gap,
            -col_gap,
            len(col_candidate.row_bands),
            len(col_candidate.answer_bands),
            row_candidate.gap_close_ratio,
            row_candidate.matched_row_ratio,
            gap_score,
        )

    if layout == "row":
        return (
            3 if row_confident else 2,
            row_candidate.gap_close_ratio,
            row_candidate.matched_row_ratio,
            gap_score,
            row_candidate.num_cols,
            len(row_candidate.bands),
            -abs(len(col_candidate.row_bands) - EXPECTED_ROWS) if EXPECTED_ROWS > 0 else 0,
            -abs(len(col_candidate.answer_bands) - EXPECTED_ANSWER_COLS)
            if EXPECTED_ANSWER_COLS > 0
            else 0,
        )

    return (
        0,
        row_candidate.gap_close_ratio,
        row_candidate.matched_row_ratio,
        gap_score,
        len(col_candidate.row_bands),
        len(col_candidate.answer_bands),
        int(row_candidate.valid),
        int(col_candidate.valid),
    )


def _question_start_for_page(stem, layout, num_cols, row_count, global_state):
    if GLOBAL_NUMBERING:
        start = global_state["current"]
        global_state["current"] += num_cols * row_count
        return start

    if layout == "row":
        return ROW_PAGE_START_QUESTION.get(stem, 1)
    return COL_PAGE_START_QUESTION.get(stem, 1)


def _render_pdf_page(page):
    scale = PDF_RENDER_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _pdf_page_stem(pdf_path, page_number, page_count):
    if page_count <= 1:
        return pdf_path.stem
    return f"{pdf_path.stem}_pg_{page_number}"


def _pdf_numbering_stem(pdf_path, page_number, page_count):
    page_stem = f"{pdf_path.stem}_pg_{page_number}"
    if page_count > 1:
        return page_stem

    if (
        pdf_path.stem in ROW_PAGE_START_QUESTION
        or pdf_path.stem in COL_PAGE_START_QUESTION
    ):
        return pdf_path.stem
    if (
        page_stem in ROW_PAGE_START_QUESTION
        or page_stem in COL_PAGE_START_QUESTION
    ):
        return page_stem
    return pdf_path.stem


def _load_input_pages():
    inputs = [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}
    ]

    pages = []
    for input_path in sorted(inputs, key=lambda path: path.name.lower()):
        if input_path.suffix.lower() == ".pdf":
            try:
                doc = fitz.open(input_path)
            except Exception as exc:
                print(f"Skipped (could not read PDF): {input_path} ({exc})")
                continue

            with doc:
                page_count = doc.page_count
                for page_index in range(page_count):
                    page_number = page_index + 1
                    output_stem = _pdf_page_stem(input_path, page_number, page_count)
                    numbering_stem = _pdf_numbering_stem(
                        input_path,
                        page_number,
                        page_count,
                    )
                    display_name = (
                        input_path.name
                        if page_count == 1
                        else f"{input_path.name}[page {page_number}]"
                    )
                    pages.append(
                        InputPage(
                            display_name,
                            output_stem,
                            numbering_stem,
                            _render_pdf_page(doc.load_page(page_index)),
                        )
                    )
            continue

        img = cv2.imread(str(input_path))
        if img is None:
            print("Skipped (could not read):", input_path)
            continue

        pages.append(InputPage(input_path.name, input_path.stem, input_path.stem, img))

    return pages


def _crop_row_layout(img, page, candidate, csv_rows, global_state):
    img_h, img_w = img.shape[:2]
    start_q = _question_start_for_page(
        page.numbering_stem,
        "row",
        candidate.num_cols,
        len(candidate.bands),
        global_state,
    )

    out_dir = OUTPUT_DIR / page.output_stem
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
                ["row", page.display_name, r_idx + 1, c_idx + 1, q_num, str(out_path)]
            )
            cv2.rectangle(debug, (x0p, y0p), (x1p, y1p), (0, 255, 0), 1)

    return debug


def _crop_col_layout(img, page, candidate, csv_rows, global_state):
    img_h, img_w = img.shape[:2]
    row_bands = sorted(candidate.row_bands, key=lambda band: band[0])
    answer_bands = sorted(candidate.answer_bands, key=lambda band: band[0])
    row_count = len(row_bands)
    start_q = _question_start_for_page(
        page.numbering_stem,
        "col",
        len(answer_bands),
        row_count,
        global_state,
    )

    out_dir = OUTPUT_DIR / page.output_stem
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
                ["col", page.display_name, r_idx + 1, c_idx + 1, q_num, str(out_path)]
            )
            cv2.rectangle(debug, (x0p, y0p), (x1p, y1p), (0, 255, 0), 1)

    return debug


def process_page(page, csv_rows, global_state):
    img, preprocess_info = _preprocess_page_image(page.image)
    deskew_angle = 0.0
    if preprocess_info["perspective_quad"] is not None:
        print(f"Perspective-corrected {page.display_name}")

    (
        img,
        gray,
        bw,
        pre_roi_img,
        grid_roi,
        grid_align_angle,
        grid_perspective_quad,
    ) = _prepare_page_for_layout(img)
    if grid_perspective_quad is None and DESKEW:
        deskewed_img, deskew_angle = _deskew_image(pre_roi_img)
        if abs(deskew_angle) > 0.01:
            print(f"Deskewed {page.display_name}: {deskew_angle:.2f} deg")
            (
                img,
                gray,
                bw,
                pre_roi_img,
                grid_roi,
                grid_align_angle,
                grid_perspective_quad,
            ) = _prepare_page_for_layout(
                deskewed_img,
                align_grid=False,
            )

    img_h, img_w = img.shape[:2]
    if grid_perspective_quad is not None:
        print(f"Grid-warped {page.display_name}")
    if abs(grid_align_angle) > 0.0:
        print(f"Grid-aligned {page.display_name}: {grid_align_angle:.2f} deg")

    row_candidate = _build_row_candidate(bw, img_h, img_w)
    col_candidate = _build_col_candidate(bw, img_h, img_w)
    layout = _choose_layout(row_candidate, col_candidate)

    if layout is None:
        print("No supported layout found:", page.display_name)
        return

    dbg_dir = DEBUG_DIR / page.output_stem
    dbg_dir.mkdir(parents=True, exist_ok=True)
    if preprocess_info["perspective_quad"] is not None:
        perspective_debug = page.image.copy()
        quad = np.round(preprocess_info["perspective_quad"]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(perspective_debug, [quad], True, (0, 255, 255), 3)
        cv2.imwrite(str(dbg_dir / "perspective_detected.png"), perspective_debug)
        cv2.imwrite(str(dbg_dir / "rectified.png"), preprocess_info["rectified_image"])
    if grid_roi is not None:
        x0, y0, x1, y1 = grid_roi
        grid_debug = pre_roi_img.copy()
        cv2.rectangle(grid_debug, (x0, y0), (x1, y1), (255, 180, 0), 3)
        cv2.imwrite(str(dbg_dir / "grid_roi.png"), grid_debug)
    if grid_perspective_quad is not None:
        cv2.imwrite(str(dbg_dir / "grid_perspective.png"), img)
    if abs(grid_align_angle) > 0.0:
        cv2.imwrite(str(dbg_dir / "grid_aligned.png"), img)
    if abs(deskew_angle) > 0.0:
        cv2.imwrite(str(dbg_dir / "deskew.png"), pre_roi_img)

    if layout == "row":
        debug = _crop_row_layout(img, page, row_candidate, csv_rows, global_state)
        cv2.imwrite(str(dbg_dir / "cells_row.png"), debug)
        print(
            f"Done: {page.display_name} -> layout=row "
            f"({len(row_candidate.bands)} rows x {row_candidate.num_cols} cols, "
            f"gap_cv={row_candidate.gap_cv:.3f}, "
            f"uniform_gap_ratio={row_candidate.gap_close_ratio:.3f})"
        )
        return

    debug = _crop_col_layout(img, page, col_candidate, csv_rows, global_state)
    cv2.imwrite(str(dbg_dir / "cells_col.png"), debug)
    print(
        f"Done: {page.display_name} -> layout=col "
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

    pages = _load_input_pages()

    if not pages:
        print("No images or PDFs found in:", INPUT_DIR)
        return

    csv_rows = [["layout", "page", "row", "col", "question", "path"]]
    global_state = {"current": GLOBAL_START}

    for page in pages:
        process_page(page, csv_rows, global_state)

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
