# Mini Project 4 — Panorama Stitching (Two Images)

This project stitches **two overlapping images** into a single panorama using a classic feature-based pipeline:
**ORB → matching → RANSAC homography → warping → blending/cropping**.

The input images (`imgA.png`, `imgB.png`) are **captured by me** (phone camera). I intentionally kept high overlap (~75–80%) and similar lighting to improve matching stability.

---

## Files

- `stitch.py` — main stitching script (two-image panorama)
- `imgA.png` — input image A (left-ish view)
- `imgB.png` — input image B (right-ish view)
- `panorama.png` — output panorama (generated after running the script)

---

## Environment / Dependencies

- Python 3
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy

If you are using Conda, you can also install OpenCV with:

conda install -c conda-forge opencv numpy
```

⸻

How to Run
	1.	Put stitch.py, imgA.png, imgB.png in the same folder.
	2.	Run:

python stitch.py

	3.	The script will generate:

	•	panorama.png (stitched output)
	•	and also display the panorama in a window.

⸻

Capture Method / Image Choice (My Input)

To make stitching work reliably, I captured images with:
	•	High overlap (~75–80%) between the two frames
	•	Mostly horizontal camera motion (no zoom)
	•	Similar exposure/lighting between frames
	•	A scene containing stable edges/corners (e.g., window frame / curtain / furniture), which provides strong ORB keypoints.

⸻

Stitching Pipeline (Implementation Summary)
	1.	Feature detection & description (ORB)
Detect keypoints and compute binary descriptors on both images.
	2.	Feature matching (BFMatcher + KNN + ratio test)
Use KNN matching and apply Lowe’s ratio test to remove ambiguous matches.
	3.	Robust homography estimation (RANSAC)
Estimate the homography matrix H using cv2.findHomography(..., RANSAC, ...) to reject outlier matches.
	4.	Warping onto a larger canvas
Warp one image into the other image’s coordinate frame using cv2.warpPerspective.
A larger canvas is computed by projecting image corners and applying translation to avoid clipping.
	5.	Blending + cropping
Blend in the overlap area using masks (feathering / weighted blending) to reduce visible seams,
then crop black borders to get a cleaner panorama.

⸻

Output

The final panorama is saved as:
	•	panorama.png

⸻

Common Failure Cases
	•	Too little overlap → not enough matches / homography fails
	•	Low-texture regions (blank wall / sky) → few stable keypoints
	•	Highly repetitive patterns or symmetry → wrong matches, distorted warp
	•	Large viewpoint change (moving closer/farther) → parallax, misalignment
	•	Motion blur / moving objects (wind moving leaves, people walking) → unstable features

⸻

Notes
	•	If the panorama direction looks wrong (canvas shifted strangely), swapping the input order (A/B) often fixes it.
	•	Higher overlap generally increases the number of inliers and makes RANSAC more stable.
