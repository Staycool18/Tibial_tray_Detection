import argparse
import cv2
import numpy as np
import pandas as pd
import os


def load_mask(image_path, thresh=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    _, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return bw


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest


def contour_mask_from_contour(shape, contour):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def extract_vertical_spans(mask, bbox):
    x, y, w, h = bbox
    vertical_rows = []
    for col in range(x, x + w):
        ys = np.where(mask[:, col] > 0)[0]
        if ys.size:
            top = int(ys.min())
            bottom = int(ys.max())
            vertical_rows.append((col, top, bottom, bottom - top + 1))
        else:
            vertical_rows.append((col, None, None, 0))
    return vertical_rows


def extract_horizontal_spans(mask, bbox):
    x, y, w, h = bbox
    horizontal_cols = []
    for row in range(y, y + h):
        xs = np.where(mask[row, :] > 0)[0]
        # restrict to bbox horizontally for clarity
        xs = xs[(xs >= x) & (xs < x + w)]
        if xs.size:
            left = int(xs.min())
            right = int(xs.max())
            horizontal_cols.append((row, left, right, right - left + 1))
        else:
            horizontal_cols.append((row, None, None, 0))
    return horizontal_cols


def visualize_spans(img_path, mask, vertical, horizontal, out_path):
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # draw vertical top/bottom
    for col, top, bottom, _ in vertical:
        if top is None:
            continue
        cv2.circle(color, (col, top), 1, (0, 255, 0), -1)
        cv2.circle(color, (col, bottom), 1, (0, 0, 255), -1)
    # draw horizontal left/right
    for row, left, right, _ in horizontal:
        if left is None:
            continue
        cv2.circle(color, (left, row), 1, (255, 0, 0), -1)
        cv2.circle(color, (right, row), 1, (0, 255, 255), -1)
    cv2.imwrite(out_path, color)


def main():
    parser = argparse.ArgumentParser(description="Extract per-column height and per-row width pixels from a white shape on black background.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--thresh", type=int, default=127, help="Threshold to binarize (0-255)")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mask = load_mask(args.image, thresh=args.thresh)
    contour = find_largest_contour(mask)
    if contour is None:
        raise RuntimeError("No white shapes found in the image")

    bbox = cv2.boundingRect(contour)
    filled = contour_mask_from_contour(mask.shape, contour)

    vertical = extract_vertical_spans(filled, bbox)
    horizontal = extract_horizontal_spans(filled, bbox)

    # Save CSVs
    vdf = pd.DataFrame(vertical, columns=["x", "top_y", "bottom_y", "height"])
    hdf = pd.DataFrame(horizontal, columns=["y", "left_x", "right_x", "width"])
    vcsv = os.path.join(args.out_dir, "vertical_spans.csv")
    hcsv = os.path.join(args.out_dir, "horizontal_spans.csv")
    vdf.to_csv(vcsv, index=False)
    hdf.to_csv(hcsv, index=False)

    vis_path = os.path.join(args.out_dir, "spans_visualization.png")
    visualize_spans(args.image, mask, vertical, horizontal, vis_path)

    print(f"Saved vertical spans -> {vcsv}")
    print(f"Saved horizontal spans -> {hcsv}")
    print(f"Saved visualization -> {vis_path}")


if __name__ == "__main__":
    main()
