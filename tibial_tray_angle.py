"""
tibial_tray_angle.py

Finds the rotation angle of a tibial tray's main (long) axis, using PCA
instead of cv2.minAreaRect's angle -- because minAreaRect's angle convention
is confusing (it can silently swap which side is "width" vs "height" and
flips its reference depending on OpenCV version), whereas PCA gives one
clean, well-defined number tied directly to the actual pixel distribution.

SIGN CONVENTION (validated against known rotations before trusting it):
    0 deg   = long axis perfectly horizontal
    POSITIVE angle = tray rotated CLOCKWISE (as you look at the image)
    NEGATIVE angle = tray rotated ANTICLOCKWISE (counter-clockwise)
    Range reported: -90 deg (exclusive) to +90 deg (inclusive)
    (a line has no head/tail, so anything past +-90 deg just wraps back
    around to the same physical axis -- e.g. +95 deg is the same axis as -85 deg)

How it works:
    1. Segment the tray (Otsu + corner-polarity check, same as before)
    2. Take EVERY foreground pixel's (x, y) coordinate -- not just the
       contour -- and run PCA (eigen-decomposition of the covariance matrix)
    3. The eigenvector with the LARGEST eigenvalue points along the
       direction the pixels are most spread out in -- i.e. the tray's long
       axis
    4. atan2(vy, vx) on that eigenvector, in image pixel coordinates
       (x = right, y = down), already gives a clockwise-positive angle for
       free -- because the y-axis is flipped (points down) compared to
       normal math convention, the usual counter-clockwise-positive atan2
       result flips into clockwise-positive automatically. No manual sign
       flip is needed; this was verified numerically against rectangles
       rotated by known angles before being used here.

Usage:
    python tibial_tray_angle.py pic17.png
    python tibial_tray_angle.py pic7.jpeg pic11.jpeg
"""

import sys
import cv2
import numpy as np


def segment_object(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = otsu.shape
    corner_vals = [otsu[0, 0], otsu[0, w - 1], otsu[h - 1, 0], otsu[h - 1, w - 1]]
    bg_value = max(set(corner_vals), key=corner_vals.count)
    mask = cv2.bitwise_not(otsu) if bg_value == 255 else otsu
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def principal_axis_angle(mask):
    """Returns (angle_deg, center_xy, principal_vec, minor_vec)."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    center = pts.mean(axis=0)
    pts_centered = pts - center

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascending eigenvalue order
    principal_vec = eigvecs[:, np.argmax(eigvals)]  # long axis (largest spread)
    minor_vec = eigvecs[:, np.argmin(eigvals)]       # short axis

    vx, vy = principal_vec
    angle_deg = np.degrees(np.arctan2(vy, vx))
    angle_deg = ((angle_deg + 90) % 180) - 90       # normalize to (-90, 90]

    return angle_deg, center, principal_vec, minor_vec


def measure_angle(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = segment_object(gray)

    angle_deg, center, principal_vec, minor_vec = principal_axis_angle(mask)

    if abs(angle_deg) < 0.5:
        direction = "aligned (no meaningful tilt)"
    elif angle_deg > 0:
        direction = f"{angle_deg:.1f} deg CLOCKWISE from horizontal"
    else:
        direction = f"{abs(angle_deg):.1f} deg ANTICLOCKWISE from horizontal"

    print(f"\n=== {image_path} ===")
    print(f"principal axis angle : {angle_deg:.2f} deg")
    print(f"interpretation        : {direction}")

    # --- Annotate: principal axis line, horizontal reference line, center ---
    vis = img.copy()
    cx, cy = center
    L = 120  # half-length of the drawn axis line, in pixels

    p1 = (int(cx - principal_vec[0] * L), int(cy - principal_vec[1] * L))
    p2 = (int(cx + principal_vec[0] * L), int(cy + principal_vec[1] * L))
    cv2.line(vis, p1, p2, (255, 0, 255), 2)          # magenta = principal (long) axis

    cv2.line(vis, (int(cx - L), int(cy)), (int(cx + L), int(cy)), (0, 255, 255), 1)  # yellow = horizontal reference

    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # red = center
    cv2.putText(vis, f"{angle_deg:+.1f} deg", (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    out_path = image_path.rsplit(".", 1)[0] + "_angle.png"
    cv2.imwrite(out_path, vis)
    print(f"Annotated image saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tibial_tray_angle.py <image1> [image2] ...")
        sys.exit(1)
    for path in sys.argv[1:]:
        measure_angle(path)