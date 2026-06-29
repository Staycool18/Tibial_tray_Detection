import cv2
import numpy as np

def rotate_image(image_path, angle, output_path):
    # Read image
    image = cv2.imread(image_path)

    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # white background
    )

    # Save result
    cv2.imwrite(output_path, rotated)

# Example
rotate_image("pic17.png", -1, "rotatedright1.jpg")