import cv2
import numpy as np


def orb_match(imgA, imgB, max_features=3000, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=max_features)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    kpA, desA = orb.detectAndCompute(grayA, None)
    kpB, desB = orb.detectAndCompute(grayB, None)

    if desA is None or desB is None or len(kpA) < 8 or len(kpB) < 8:
        return None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desB, desA, k=2)  # map B -> A

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, None, len(good)

    ptsB = np.float32([kpB[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # in new
    ptsA = np.float32([kpA[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # in base
    return ptsA, ptsB, len(good)


def make_canvas_and_warp(base, new_img, H):
    h1, w1 = base.shape[:2]
    h2, w2 = new_img.shape[:2]

    corners_base = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_new = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    warped_corners_new = cv2.perspectiveTransform(corners_new, H)
    all_corners = np.vstack((corners_base, warped_corners_new))

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0

    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)

    out_w = int(x_max - x_min)
    out_h = int(y_max - y_min)

    warped_new = cv2.warpPerspective(new_img, T @ H, (out_w, out_h))

    canvas = warped_new.copy()
    canvas[ty:ty + h1, tx:tx + w1] = base
    return canvas


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    nz = cv2.findNonZero(th)
    if nz is None:
        return img
    x, y, w, h = cv2.boundingRect(nz)
    return img[y:y + h, x:x + w]


def stitch_two(img_left, img_right, ransac_thresh=5.0):

    ptsA, ptsB, n_good = orb_match(img_left, img_right)
    if ptsA is None:
        raise RuntimeError(f"Not enough matches. good_matches={n_good}")

    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("findHomography failed")

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 8:
        raise RuntimeError(f"Too few inliers: {inliers}")

    pano = make_canvas_and_warp(img_left, img_right, H)
    pano = crop_black(pano)
    return pano, n_good, inliers


if __name__ == "__main__":
    img1_path = "imgA.png"
    img2_path = "imgB.png"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Image path wrong. Check img1_path/img2_path.")

    pano, good, inliers = stitch_two(img1, img2)
    print(f"OK. good_matches={good}, inliers={inliers}")

    cv2.imwrite("panorama.png", pano)
    cv2.imshow("Panorama", pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()