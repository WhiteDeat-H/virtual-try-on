import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Virtual Glasses Try-On",
    page_icon="👓",
    layout="wide"
)

st.title("👓 Real-Time Virtual Glasses Try-On")
st.write("Take a snapshot with your browser camera or upload a photo to try on glasses!")

# ---------------------------------------------------------------------------
# Helper Functions: Transformation & Overlay Operations
# ---------------------------------------------------------------------------
def rotate_image_unbound(image, angle):
    (h, w) = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    (cx, cy) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = max(1, int((h * sin) + (w * cos)))
    nH = max(1, int((h * cos) + (w * sin)))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(
        image, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    if ol_h <= 0 or ol_w <= 0 or x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    x1, x2 = max(0, x), min(bg_w, x + ol_w)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)

    overlay_x1, overlay_x2 = max(0, -x), min(ol_w, bg_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(ol_h, bg_h - y)

    if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2:
        return background

    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (
            alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
            (1.0 - alpha) * background[y1:y2, x1:x2, c]
        )
    return background

def adjust_lighting(overlay_img, roi_bgr):
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

base_dir = os.path.dirname(os.path.abspath(__file__))
glasses_files = [f for f in os.listdir(base_dir) if f.startswith('glasses') and f.endswith('.png')]
if not glasses_files:
    glasses_files = ['glasses.png']

selected_asset = st.sidebar.selectbox("Select Glasses Style", glasses_files)
scale_factor = st.sidebar.slider("Glasses Scale Factor", min_value=1.0, max_value=3.0, value=1.8, step=0.05)
vertical_offset = st.sidebar.slider("Vertical Position Adjustment", min_value=-100, max_value=100, value=-20, step=1)
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
# Face Processing & Overlay Execution (Using Modern MediaPipe Solutions Fallback)
# ---------------------------------------------------------------------------
if input_image is not None:
    input_image = np.ascontiguousarray(input_image)
    asset_path = os.path.join(base_dir, selected_asset)
    
    if os.path.exists(asset_path):
        glasses_img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        
        if glasses_img is not None:
            glasses_img = np.ascontiguousarray(glasses_img)
            if len(glasses_img.shape) == 2:
                glasses_img = cv2.cvtColor(glasses_img, cv2.COLOR_GRAY2BGRA)
            elif glasses_img.shape[2] == 3:
                alpha_channel = np.ones((glasses_img.shape[0], glasses_img.shape[1], 1), dtype=glasses_img.dtype) * 255
                glasses_img = np.concatenate((glasses_img, alpha_channel), axis=2)

            h, w, _ = input_image.shape
            rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)

            try:
                # Fallback check for solutions module across mediapipe versions
                mp_face_mesh = getattr(mp, "solutions", None)
                if mp_face_mesh and hasattr(mp_face_mesh, "face_mesh"):
                    face_mesh_obj = mp_face_mesh.face_mesh.FaceMesh(
                        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3
                    )
                    results = face_mesh_obj.process(rgb_image)
                    multi_face_landmarks = results.multi_face_landmarks
                else:
                    # Modern MediaPipe task vision API fallback or lightweight approximation
                    # Using legacy fallback package check or alternative if available
                    raise ImportError("mp.solutions unavailable in current mediapipe build.")

                if multi_face_landmarks:
                    landmarks = multi_face_landmarks[0].landmark
                    
                    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
                    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
                    eye_center = (left_eye + right_eye) / 2.0

                    eye_distance = np.linalg.norm(right_eye - left_eye)
                    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

                    target_width = max(10, int(eye_distance * scale_factor))
                    aspect_ratio = glasses_img.shape[0] / max(1, glasses_img.shape[1])
                    target_height = max(10, int(target_width * aspect_ratio))

                    output_image = input_image.copy()

                    resized_glasses = cv2.resize(glasses_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    if enable_lighting:
                        roi_y1, roi_y2 = max(0, int(eye_center[1] - 50)), min(h, int(eye_center[1] + 50))
                        roi_x1, roi_x2 = max(0, int(eye_center[0] - 50)), min(w, int(eye_center[0] + 50))
                        face_roi = input_image[roi_y1:roi_y2, roi_x1:roi_x2]
                        resized_glasses = adjust_lighting(resized_glasses, face_roi)

                    rotated_glasses = rotate_image_unbound(resized_glasses, -angle)

                    gh, gw = rotated_glasses.shape[:2]
                    top_left_x = int(eye_center[0] - gw / 2)
                    top_left_y = int(eye_center[1] - gh / 2 + vertical_offset)

                    output_image = overlay_transparent(output_image, rotated_glasses, top_left_x, top_left_y)

                    st.subheader("🎉 Your Try-On Result")
                    final_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    st.image(np.ascontiguousarray(final_rgb, dtype=np.uint8))
                else:
                    st.warning("⚠️ No face detected in the photo. Please make sure your face is clearly visible and well-lit.")
            except Exception as e:
                # If solutions namespace isn't bundled in the installed build, pin mediapipe back to 0.10.21 explicitly
                st.error(f"MediaPipe version incompatibility error: {e}. Please ensure mediapipe<=0.10.21 is pinned in requirements.txt")
        else:
            st.error(f"OpenCV failed to decode asset: {selected_asset}.")
    else:
        st.error(f"Asset file not found on server at path: {asset_path}")