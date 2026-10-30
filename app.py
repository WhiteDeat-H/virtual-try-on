import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Virtual Glasses Try-On",
    page_icon="👓",
    layout="wide"
)

st.title("👓 Real-Time Virtual Glasses Try-On")
st.write("Take a snapshot with your camera or upload a photo to try on glasses in real-time!")

# ---------------------------------------------------------------------------
# Helper Functions: Transformation & Overlay Operations
# ---------------------------------------------------------------------------
def rotate_image_unbound(image, angle):
    """
    Rotates an image by an angle without clipping corners by expanding the canvas.
    """
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(
        image, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def overlay_transparent(background, overlay, x, y):
    """
    Blends a 4-channel RGBA overlay onto a 3-channel BGR background image at (x, y).
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    x1, x2 = max(0, x), min(bg_w, x + ol_w)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)

    overlay_x1, overlay_x2 = max(0, -x), min(ol_w, bg_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(ol_h, bg_h - y)

    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (
            alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
            (1.0 - alpha) * background[y1:y2, x1:x2, c]
        )
    return background

def adjust_lighting(overlay_img, roi_bgr):
    """
    Dynamically scales overlay asset brightness matching facial ROI.
    """
    if roi_bgr.size == 0:
        return overlay_img
    
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv_roi[:, :, 2])
    scale = np.clip(avg_brightness / 128.0, 0.6, 1.3)
    
    adjusted = overlay_img.astype(np.float32)
    adjusted[:, :, :3] *= scale
    adjusted[:, :, :3] = np.clip(adjusted[:, :, :3], 0, 255)
    return adjusted.astype(np.uint8)

# ---------------------------------------------------------------------------
# Sidebar UI Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Try-On Settings")

glasses_files = [f for f in os.listdir('.') if f.startswith('glasses') and f.endswith('.png')]
if not glasses_files:
    glasses_files = ['glasses.png']

selected_asset = st.sidebar.selectbox("Select Glasses Style", glasses_files)
scale_factor = st.sidebar.slider("Glasses Scale Factor", min_value=1.0, max_value=2.5, value=1.6, step=0.05)
enable_lighting = st.sidebar.checkbox("Enable Dynamic Lighting", value=True)

# ---------------------------------------------------------------------------
# Input Controls & Frame Capture
# ---------------------------------------------------------------------------
input_mode = st.radio("Choose Input Mode:", ["Use Web Camera", "Upload Photo"], horizontal=True)

input_image = None

if input_mode == "Use Web Camera":
    camera_file = st.camera_input("Take a snapshot to try on glasses")
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    uploaded_file = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ---------------------------------------------------------------------------
# Face Processing & Overlay Execution
# ---------------------------------------------------------------------------
if input_image is not None and os.path.exists(selected_asset):
    glasses_img = cv2.imread(selected_asset, cv2.IMREAD_UNCHANGED)
    
    if glasses_img is not None and glasses_img.shape[2] == 4:
        h, w, _ = input_image.shape
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Access MediaPipe FaceMesh safely inside context manager
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Key landmarks: 33 (Left Eye Outer), 263 (Right Eye Outer), 6 (Nose Bridge)
                left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
                right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
                nose_bridge = np.array([landmarks[6].x * w, landmarks[6].y * h])

                eye_distance = np.linalg.norm(right_eye - left_eye)
                angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

                target_width = int(eye_distance * scale_factor)
                aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
                target_height = int(target_width * aspect_ratio)

                if target_width > 0 and target_height > 0:
                    resized_glasses = cv2.resize(glasses_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    if enable_lighting:
                        roi_y1, roi_y2 = max(0, int(nose_bridge[1] - 50)), min(h, int(nose_bridge[1] + 50))
                        roi_x1, roi_x2 = max(0, int(nose_bridge[0] - 50)), min(w, int(nose_bridge[0] + 50))
                        face_roi = input_image[roi_y1:roi_y2, roi_x1:roi_x2]
                        resized_glasses = adjust_lighting(resized_glasses, face_roi)

                    rotated_glasses = rotate_image_unbound(resized_glasses, -angle)

                    gh, gw = rotated_glasses.shape[:2]
                    top_left_x = int(nose_bridge[0] - gw / 2)
                    top_left_y = int(nose_bridge[1] - gh / 2)

                    output_image = input_image.copy()
                    output_image = overlay_transparent(output_image, rotated_glasses, top_left_x, top_left_y)

                    st.subheader("🎉 Your Try-On Result")
                    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.warning("No face detected in the photo. Please ensure your face is clearly visible and well-lit.")
    else:
        st.error(f"Asset format error in {selected_asset}. Please ensure it is a 4-channel transparent PNG file.")