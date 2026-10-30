![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00979D?style=for-the-badge&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---
# Real-Time Virtual Try-On (VTO) Application

An interactive Real-Time Virtual Try-On application built with Python, OpenCV, MediaPipe, and Streamlit. It tracks facial landmarks in real time to accurately overlay eyewear accessories with head-tilt compensation, jitter reduction, ambient lighting adjustment, and soft 3D drop shadows.

---

## Key Features

* **Facial Landmark Tracking:** Uses MediaPipe Face Mesh to dynamically map eye distance and nose bridge positioning.
* **Head-Tilt & Rotation Compensation:** Employs an affine rotation matrix with dynamic canvas padding to prevent corner clipping during head movement.
* **Jitter Reduction:** Applies Exponential Moving Average (EMA) smoothing to stabilize asset placement across video frames.
* **3D Depth & Ambient Lighting:** Computes face region brightness to dynamically adjust frame contrast and renders a soft Gaussian drop shadow.
* **Multi-Asset Cycling & Snapshots:** Cycle through multiple glasses frames (`N`/`P` keys) and capture instant snapshots (`S` key).
* **Dual Interface:** Runs as both a native OpenCV desktop app (`main.py`) and a web interface (`app.py` via Streamlit).

---

## Project Structure

```text
virtual-try-on/
│
├── main.py             # Desktop OpenCV VTO application
├── app.py              # Web-based Streamlit dashboard
├── glasses.png         # Eyewear PNG asset (Transparent)
├── glasses1.png        # Eyewear PNG asset (Transparent)
├── glasses2.png        # Eyewear PNG asset (Transparent)
├── glasses3.png        # Eyewear PNG asset (Transparent)
├── requirements.txt    # Project dependencies
└── .gitignore          # Excluded tracking files