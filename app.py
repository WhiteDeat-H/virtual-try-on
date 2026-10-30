import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os

st.set_page_config(page_title="Virtual Glasses Try-On", layout="wide")

st.title("🕶️ Real-Time Virtual Glasses Try-On")
st.sidebar.header("Controls & Assets")

# Load available glasses options
glasses_files = ['glasses.png', 'glasses1.png', 'glasses2.png', 'glasses3.png']
available_glasses = [f for f in glasses_files if os.path.exists(f)]

if not available_glasses:
    st.error("No transparent PNG glasses found in the project folder!")
    st.stop()

selected_glasses_file = st.sidebar.selectbox("Select Glasses Frame", available_glasses)
scale_factor = st.sidebar.slider("Adjust Frame Size", 1.2, 2.5, 1.9, 0.1)

def rotate_image_unbound(image, angle):
    """Rotates image with canvas expansion to avoid corner clipping."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1, y2 = max(0, y), min(bg_h, y + h)
    overlay_x1, overlay_x2 = max(0, -x), w - max(0, (x + w) - bg_w)
    overlay_y1, overlay_y2 = max(0, -y), h - max(0, (y + h) - bg_h)
    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    for color in range(3):
        background[y1:y2, x1:x2, color] = (
            alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, color] +
            (1.0 - alpha) * background[y1:y2, x1:x2, color]
        )
    return background

run_app = st.checkbox("Turn On Webcam", value=True)
FRAME_WINDOW = st.image([])

glasses_img = cv2.imread(selected_glasses_file, cv2.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Landmark Smoothing Variables
prev_left_eye = None
prev_right_eye = None
prev_nose = None
alpha_smooth = 0.6

while run_app and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Unable to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            curr_left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
            curr_right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
            curr_nose = np.array([face_landmarks.landmark[6].x * w, face_landmarks.landmark[6].y * h])

            if prev_left_eye is None:
                left_eye, right_eye, nose_bridge = curr_left_eye, curr_right_eye, curr_nose
            else:
                left_eye = alpha_smooth * curr_left_eye + (1 - alpha_smooth) * prev_left_eye
                right_eye = alpha_smooth * curr_right_eye + (1 - alpha_smooth) * prev_right_eye
                nose_bridge = alpha_smooth * curr_nose + (1 - alpha_smooth) * prev_nose

            prev_left_eye, prev_right_eye, prev_nose = left_eye, right_eye, nose_bridge

            eye_distance = np.linalg.norm(right_eye - left_eye)
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            glasses_width = int(eye_distance * scale_factor)
            glasses_height = int(glasses_width * (glasses_img.shape[0] / glasses_img.shape[1]))

            if glasses_width > 0 and glasses_height > 0:
                resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
                rotated_glasses = rotate_image_unbound(resized_glasses, -angle)

                rot_h, rot_w, _ = rotated_glasses.shape
                x_center = int(nose_bridge[0])
                y_center = int(nose_bridge[1])

                top_left_x = x_center - int(rot_w / 2)
                top_left_y = y_center - int(rot_h / 2)

                frame = overlay_transparent(frame, rotated_glasses, top_left_x, top_left_y)
    else:
        prev_left_eye = prev_right_eye = prev_nose = None

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()