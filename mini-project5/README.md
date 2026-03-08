# Mini Project 5: Lane Detector

This project is a simple lane detector made with OpenCV and NumPy.  
It takes a road image or video, detects the lane lines, and draws the final left and right lane boundaries in red.

## Team members
- Mingzhe Ou
- Yi-hsuan Lai
- Jia Song

## Files
- `lane_detector.py` → main program
- `README.md` → project instructions
- `road.mp4` → input video files
- `demo.mp4` → Demon showing result from processing road.mp4
## What the program does
The program processes each frame step by step:
1. Convert image to grayscale
2. Blur the image
3. Detect edges using Canny
4. Keep only the road area using ROI
5. Use Hough Line Transform to detect line segments
6. Split lines into left lane and right lane based on slope
7. Average the lines
8. Draw the final lane lines in red

## Keyboard Controls
- **q** — Quit application

## Requirements
Install these first:

```bash
pip install opencv-python numpy