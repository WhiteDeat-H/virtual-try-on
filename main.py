import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# List of glasses files to cycle through
glasses_files = ['glasses.png', 'glasses1.png', 'glasses2.png', 'glasses3.png']
glasses_images = []

for file in glasses_files:
    if os.path.exists(file):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            glasses_images.append((file, img))

if not glasses_images:
    print("Error: No valid transparent PNG glasses assets found in folder.")
    exit()

current_idx = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def apply_dynamic_lighting(glasses_img, face_bbox_roi):
    """Adjusts brightness of the glasses based on ambient facial ROI lighting."""
    if face_bbox_roi.size == 0 or glasses_img is None:
        return glasses_img

    hsv_face = cv2.cvtColor(face_bbox_roi, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv_face[:, :, 2])

    brightness_factor = np.clip(avg_brightness / 128.0, 0.6, 1.3)

    rgb = glasses_img[:, :, :3].astype(np.float32)
    alpha = glasses_img[:, :, 3]

    adjusted_rgb = np.clip(rgb * brightness_factor, 0, 255).astype(np.uint8)

    return cv2.merge([adjusted_rgb[:, :, 0], adjusted_rgb[:, :, 1], adjusted_rgb[:, :, 2], alpha])

def overlay_with_shadow(background, overlay, x, y, shadow_offset=(3, 5), shadow_blur=11, shadow_opacity=0.35):
    """Overlays transparent PNG with an ambient Gaussian drop shadow underneath."""
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    alpha = overlay[:, :, 3] / 255.0

    shadow_mask = (alpha * 255 * shadow_opacity).astype(np.uint8)
    shadow_canvas = cv2.GaussianBlur(shadow_mask, (shadow_blur, shadow_blur), 0)

    sx, sy = x + shadow_offset[0], y + shadow_offset[1]

    # Render Shadow
    if 0 <= sx < bg_w and 0 <= sy < bg_h:
        sx1, sx2 = max(0, sx), min(bg_w, sx + w)
        sy1, sy2 = max(0, sy), min(bg_h, sy + h)

        so_x1, so_x2 = max(0, -sx), w - max(0, (sx + w) - bg_w)
        so_y1, so_y2 = max(0, -sy), h - max(0, (sy + h) - bg_h)

        s_alpha = (shadow_canvas[so_y1:so_y2, so_x1:so_x2] / 255.0)

        for c in range(3):
            background[sy1:sy2, sx1:sx2, c] = (
                (1.0 - s_alpha) * background[sy1:sy2, sx1:sx2, c]
            ).astype(np.uint8)

    # Render Overlay
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    x1, x2 = max(0, x), min(bg_w, x + w)
    y1, y2 = max(0, y), min(bg_h, y + h)

    ox1, ox2 = max(0, -x), w - max(0, (x + w) - bg_w)
    oy1, oy2 = max(0, -y), h - max(0, (y + h) - bg_h)

    ov_alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            ov_alpha * overlay[oy1:oy2, ox1:ox2, c] +
            (1.0 - ov_alpha) * background[y1:y2, x1:x2, c]
        )

    return background

prev_left_eye = None
prev_right_eye = None
prev_nose = None
alpha_smooth = 0.6

cap = cv2.VideoCapture(0)
status_message = ""
status_timer = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_filename, glasses_img = glasses_images[current_idx]

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

            glasses_width = int(eye_distance * 1.9)
            glasses_height = int(glasses_width * (glasses_img.shape[0] / glasses_img.shape[1]))

            if glasses_width > 0 and glasses_height > 0:
                resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

                min_x = max(0, int(nose_bridge[0] - glasses_width // 2))
                max_x = min(w, int(nose_bridge[0] + glasses_width // 2))
                min_y = max(0, int(nose_bridge[1] - glasses_height // 2))
                max_y = min(h, int(nose_bridge[1] + glasses_height // 2))
                face_roi = frame[min_y:max_y, min_x:max_x]

                lit_glasses = apply_dynamic_lighting(resized_glasses, face_roi)
                rotated_glasses = rotate_image_unbound(lit_glasses, -angle)

                rot_h, rot_w, _ = rotated_glasses.shape
                x_center = int(nose_bridge[0])
                y_center = int(nose_bridge[1])

                top_left_x = x_center - int(rot_w / 2)
                top_left_y = y_center - int(rot_h / 2)

                frame = overlay_with_shadow(frame, rotated_glasses, top_left_x, top_left_y)
    else:
        prev_left_eye = prev_right_eye = prev_nose = None

    cv2.putText(frame, f"Model: {current_filename} (N: Next | P: Prev | S: Save | Q: Quit)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if status_timer > 0:
        cv2.putText(frame, status_message, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        status_timer -= 1

    cv2.imshow('Virtual Glasses Try-On (3D Lighting & Shadows)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_idx = (current_idx + 1) % len(glasses_images)
    elif key == ord('p'):
        current_idx = (current_idx - 1) % len(glasses_images)
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tryon_snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        status_message = f"Saved: {filename}"
        status_timer = 30

cap.release()
cv2.destroyAllWindows()