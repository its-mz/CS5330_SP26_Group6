import cv2
import os

img = cv2.imread('LOGO1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw rich keypoints (showing size and orientation)
img_sift = cv2.drawKeypoints(gray, keypoints, img,
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save the output image
output_dir = "output"
output_path = os.path.join(output_dir, 'sift_keypoints_output.jpg')
cv2.imwrite(output_path, img_sift)

cv2.imshow('Workshop 2 - SIFT Keypoints', img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()