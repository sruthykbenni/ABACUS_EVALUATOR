import fitz
import cv2
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location('ext', 'extract_answer_boxes-auto.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pdf = fitz.open('input/answersheets_#.pdf')
page = pdf.load_page(0)
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_h, img_w = img.shape[:2]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
bw = mod._adaptive_bin(img_blur)
h_lines, v_lines = mod._detect_grid_lines(bw, img_w, img_h, mod.ROW_H_LINE_SCALE, mod.ROW_V_LINE_SCALE)
row_sum = h_lines.sum(axis=1) / 255.0
peaks = [i for i, v in enumerate(row_sum) if v > img_w * mod.ROW_LINE_THRESH_RATIO]
print('first 30 peaks', peaks[:30])
print('peak count', len(peaks))
ranges = []
for idx in peaks:
    if not ranges or idx > ranges[-1][1] + 1:
        ranges.append([idx, idx])
    else:
        ranges[-1][1] = idx
ys = [(s + e) // 2 for s, e in ranges]
print('line centers', ys[:20])
print('line center count', len(ys))
print('first 8 spans')
for i in range(min(8, len(ys) - 1)):
    y0, y1 = ys[i], ys[i + 1]
    print(i, y0, y1, y1 - y0)
ink = cv2.bitwise_and(bw, cv2.bitwise_not(cv2.bitwise_or(h_lines, v_lines)))
bands = mod._row_bands_from_lines(h_lines, ink, img_h, img_w, mod.ROW_LINE_THRESH_RATIO, mod.ROW_BAND_MIN_HEIGHT, mod.ROW_BAND_MAX_HEIGHT, mod.ROW_MIN_BAND_INK, mod.ROW_IGNORE_TOP_PCT)
print('bands', bands)
print('band count', len(bands))
for b in bands[:10]:
    print('band', b, 'height', b[1]-b[0])
