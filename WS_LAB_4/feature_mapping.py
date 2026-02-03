import cv2
import sys
import os

# 1. Load the original, clean images
img1 = cv2.imread('LOGO1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('LOGO2.jpg', cv2.IMREAD_GRAYSCALE)
if img2 is None:
    sys.exit("Could not find logo files. Ensure logo1.jpg and logo2.jpg are in WS_LAB_4.")

# 2. SIFT Detection (not 'Shift')
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. Matching (Directly using des1 and des2 variables)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)

# 4. Filter and Visualize
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

# 5. Output for Verification
print(f"Total Good Matches: {len(good_matches)}")
cv2.imshow('Workshop 3 - Feature Matching', result)
cv2.waitKey(0)

# 6. Save the output image
output_dir = "output"
output_path = os.path.join(output_dir, 'feature_matching_output.jpg')
cv2.imwrite(output_path, result)