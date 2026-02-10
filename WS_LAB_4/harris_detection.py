import cv2
import numpy as np
import os

# Load image and convert to grayscale
img = cv2.imread('blur_level_4_sigma_8.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32 for Harris detection
gray = np.float32(gray)

# Apply Harris Corner Detection
# blockSize: neighborhood size; ksize: aperture for Sobel; k: Harris parameter
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate to make corners visible
dst = cv2.dilate(dst, None)

# Threshold for marking corners in red
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Save the output image 
# output_dir = "output"
# output_path = os.path.join(output_dir, 'harris_corners_output.jpg')
# cv2.imwrite(output_path, img)

cv2.imshow('Workshop 1 - Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()