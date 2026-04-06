import cv2
import numpy as np

from crop_config import *


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


def _should_apply_perspective_warp(img, quad):
    return _quad_distortion_ratio(img.shape[:2], quad) >= PERSPECTIVE_MIN_DISTORTION_RATIO


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


__all__ = [
    "_adaptive_bin",
    "_deskew_image",
    "_normalize_line_angle",
    "_order_quad_points",
    "_preprocess_page_image",
    "_rotate_image_expand",
]
