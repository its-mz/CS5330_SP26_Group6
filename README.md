# CS5330_SP26_Group6

# Mini-project 2 Image World App (Real-Time Image Processing with OpenCV)

## Project Overview

This project is building a real-time image processing app by using **Python and OpenCV**.
The app displays a live video from the webcam, also the original view and a transformed view. It allows the users to apply different image warping and transformation effects by using the keyborad shortcuts.

## Team members

- Mingzhe Ou
- Yi-hsuan Lai
- Jia Song

## Setup and Use Instructions

- Python 3.8 or later
- OpenCV
- Git

### 1. Environment Setup

Create and activate a Python virtual environment

```bash
# Create virtual environment and activate
python -m venv webcamapp
source webcamapp/bin/activate   # For macOS / Linux
webcamapp\Scripts\activate      # For Windows
# Install dependencies
pip install opencv-python numpy
```

### 2. Run the app

```bash
python webcam.py
```

## Keyboard Controls

- **0** — Reset to original view
- **1** — Flip horizontally
- **2** — Flip vertically
- **r** — Rotate left by 5°
- **t** — Rotate right by 5°
- **+** — Scale up (+5%)
- **-** — Scale down (-5%)
- **Arrow Keys** — Translate image (up/down/left/right)
- **p** — Toggle perspective warp
- **w,a,d,x** - Perspective change
- **s** — Save screenshot
- **q** — Quit application

## Implemented Warping Techniques

- Flip: horizontal and vertical mirroring of the feed
- Rotation: ±5° rotation with the center of the feed
- Scaling: ±5% zoom in or zoom out the feed
- Translation: move the feed using the arrow keys
- Perspective Transform: viewpoint change with a four-point warp

### Features

- Real-time FPS display
- Dual views display
- Responsive controls
