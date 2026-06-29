#!/usr/bin/env python3
"""Measure implant component width and height from an image file.

This script is built to handle both light-background and dark-background
images by using Otsu thresholding plus corner-based polarity detection.
"""

import os
import sys

import cv2
import numpy as np


def segment_object(gray):
    """Segment the object from a grayscale image robustly.

    This is not a simple "white shape on black" algorithm because medical
    implant photos may have either a dark object on a light background or a
    light object on a dark background.

    We use Otsu thresholding to choose a good binary threshold automatically
    after Gaussian blur. Then we sample the 4 image corners: those corner
    pixels are usually background, so a majority vote tells us whether the
    binary background is 0 or 255. If the background is white, we invert the
    mask so the object becomes the foreground (255).

    Finally, a 3x3 closing followed by opening removes small holes and speckle
    noise while preserving the overall shape.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = thresh.shape[:2]
    corners = [
        int(thresh[0, 0]),
        int(thresh[0, w - 1]),
        int(thresh[h - 1, 0]),
        int(thresh[h - 1, w - 1]),
    ]

    # If the corners are mostly white, the background is white and the object
    # is darker. Invert so the object always becomes foreground=255.
    background_is_white = sum(1 for value in corners if value == 255) >= 2
    if background_is_white:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def compute_center(contour, mask):
    """Compute three center estimates for the detected tray shape.

    1. Axis-aligned bbox center:
       - Simple geometric center of the axis-aligned bounding rectangle.
       - This can drift if the tray is rotated, because the rectangle bulges
         around the rotated shape.

    2. Oriented bbox center:
       - Uses cv2.minAreaRect so the box rotates with the object.
       - This is the recommended center for rotated tray photos.

    3. True centroid:
       - Uses image moments on the filled mask, giving the center of mass.
       - This is included as a sanity-check; if it drifts significantly from
         the oriented bbox center, the tray shape may be unusually asymmetric
         or the rectangle approximation may be less reliable.
    """
    x, y, w, h = cv2.boundingRect(contour)
    center_aabb = (x + w / 2.0, y + h / 2.0)

    (center_oriented_x, center_oriented_y), _, _ = cv2.minAreaRect(contour)
    center_oriented = (float(center_oriented_x), float(center_oriented_y))

    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] != 0:
        center_centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        center_centroid = center_oriented

    return center_aabb, center_oriented, center_centroid


def measure(image_path):
    """Measure the largest object in the input image and save annotated output."""
    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot open image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = segment_object(gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"WARNING: No object found in {image_path}")
        return

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    (oriented_cx, oriented_cy), (oriented_w, oriented_h), angle = cv2.minAreaRect(contour)
    oriented_w = float(oriented_w)
    oriented_h = float(oriented_h)

    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    row_spans = []
    for row in range(mask.shape[0]):
        xs = np.where(mask[row, :] == 255)[0]
        if xs.size:
            row_spans.append(int(xs.max() - xs.min() + 1))
        else:
            row_spans.append(0)
    overall_width = int(max(row_spans)) if row_spans else 0

    col_spans = []
    for col in range(mask.shape[1]):
        ys = np.where(mask[:, col] == 255)[0]
        if ys.size:
            col_spans.append(int(ys.max() - ys.min() + 1))
        else:
            col_spans.append(0)
    overall_height = int(max(col_spans)) if col_spans else 0

    area_pixels = int(cv2.countNonZero(mask))
    center_aabb, center_oriented, center_centroid = compute_center(contour, mask)
    center_offset = np.hypot(center_oriented[0] - center_centroid[0], center_oriented[1] - center_centroid[1])

    print(f"\nImage: {image_path}")
    print(f"  extreme left  : {leftmost}")
    print(f"  extreme right : {rightmost}")
    print(f"  extreme top   : {topmost}")
    print(f"  extreme bottom: {bottommost}")
    print(f"  overall width (max row span)  : {overall_width}")
    print(f"  overall height (max col span) : {overall_height}")
    print(f"  axis-aligned bbox width/height: {w} x {h}")
    print(f"  oriented bbox width/height/angle: {oriented_w:.1f} x {oriented_h:.1f} @ {angle:.1f} deg")
    print(f"  total object area (pixels)   : {area_pixels}")
    print(f"  axis-aligned bbox center      : ({center_aabb[0]:.1f}, {center_aabb[1]:.1f})")
    print(f"  oriented bbox center          : ({center_oriented[0]:.1f}, {center_oriented[1]:.1f})")
    print(f"  centroid center               : ({center_centroid[0]:.1f}, {center_centroid[1]:.1f})")
    print(f"  center estimate offset        : {center_offset:.1f} pixels")

    annotated = image.copy()
    cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for point in (leftmost, rightmost, topmost, bottommost):
        cv2.circle(annotated, point, 6, (255, 0, 0), -1)
    cv2.circle(annotated, (int(round(center_aabb[0])), int(round(center_aabb[1]))), 5, (0, 0, 255), -1)
    cv2.circle(annotated, (int(round(center_oriented[0])), int(round(center_oriented[1]))), 5, (255, 0, 255), -1)
    cv2.circle(annotated, (int(round(center_centroid[0])), int(round(center_centroid[1]))), 5, (255, 255, 0), -1)

    base, _ = os.path.splitext(image_path)
    annotated_path = f"{base}_annotated.png"
    cv2.imwrite(annotated_path, annotated)
    print(f"  annotated image saved to: {annotated_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_implant_dimensions.py <image1> [image2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        measure(path)
