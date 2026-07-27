import cv2
import numpy as np
import os
import time
import mediapipe as mp

# =====================================================================
# 1. CONFIGURATION & SPECIFICATION TARGETS (From Project Workplan Spec)
# =====================================================================
STATIC_TARGET_SAMPLES = 40      # Workplan Target: 40 frames per letter
DYNAMIC_TARGET_SEQUENCES = 20   # Workplan Target: 20 videos per word
SEQUENCE_LENGTH = 30            # 30 frames per video sequence
TARGET_FPS = 30                 # Enforced execution frame rate
FRAME_DELAY = 1.0 / TARGET_FPS
MIN_VALID_FRAME_RATIO = 0.9     # At least 90% of frames in a sequence must have valid hand detection

# NEW: whether static samples must also have a valid face detection to be accepted.
# Kept as a toggle rather than hard-baked, since occasionally framing/lighting
# may cause momentary face-mesh dropout even when the sign itself is valid.
REQUIRE_FACE_FOR_STATIC = True

# Initialize MediaPipe Holistic Pipeline
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# =====================================================================
# NEW: REDUCED FACIAL FEATURE SET
# Instead of all 468 face-mesh landmarks (which would add 1404 features and
# mostly encode redundant cheek/jaw geometry), we extract only the regions
# that carry linguistic information in sign languages: eyebrows, eyes, and
# mouth/lips. This is derived programmatically from MediaPipe's own named
# connection sets, so the indices are correct and reproducible rather than
# a hand-typed guess.
# =====================================================================
_FACE_REGIONS_OF_INTEREST = [
    mp_face_mesh.FACEMESH_LIPS,
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
]
FACE_LANDMARK_INDICES = sorted({
    idx for connection_set in _FACE_REGIONS_OF_INTEREST
    for pair in connection_set
    for idx in pair
})
NUM_FACE_POINTS = len(FACE_LANDMARK_INDICES)

# Feature vector length is now computed, not hardcoded, so it's self-documenting
# and can't silently drift out of sync with the extraction logic below.
POSE_LEN = 33 * 4
LH_LEN = 21 * 3
RH_LEN = 21 * 3
FACE_LEN = NUM_FACE_POINTS * 3
FEATURE_VECTOR_LENGTH = POSE_LEN + LH_LEN + RH_LEN + FACE_LEN

print(f"[CONFIG] Reduced face landmark count: {NUM_FACE_POINTS} points "
      f"(eyebrows + eyes + lips) -> {FACE_LEN} features")
print(f"[CONFIG] Total feature vector length: {FEATURE_VECTOR_LENGTH} "
      f"(pose={POSE_LEN}, left_hand={LH_LEN}, right_hand={RH_LEN}, face={FACE_LEN})")
print("[COMPAT WARNING] This vector length differs from the previous 258-dim format. "
      "Old .npy samples are NOT compatible with this script's output and must be "
      "kept in a separate dataset folder or re-collected.\n")


def extract_keypoints(results):
    """Extracts pose, both hands, and a reduced facial landmark set (eyebrows,
    eyes, lips) into a single flattened feature vector."""
    # 1. Extract Pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_LEN)
    
    # 2. Extract Left Hand (Fixed clean conditional block)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(LH_LEN)
        
    # 3. Extract Right Hand (Fixed clean conditional block)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(RH_LEN)

    # 4. Extract Face (Reduced Set)
    if results.face_landmarks:
        all_face_lm = results.face_landmarks.landmark
        face = np.array([[all_face_lm[i].x, all_face_lm[i].y, all_face_lm[i].z]
                          for i in FACE_LANDMARK_INDICES]).flatten()
    else:
        face = np.zeros(FACE_LEN)

    return np.concatenate([pose, lh, rh, face])

def has_valid_hands(results):
    """Checks that at least one hand is visible to prevent saving empty matrices"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def has_valid_face(results):
    """Checks that the face mesh was detected at all this frame."""
    return results.face_landmarks is not None


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw full face tesselation for visual feedback during collection even though
    # only a reduced subset of points is actually saved into the feature vector.
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 200, 255), thickness=1, circle_radius=1)
        )


def build_sequence_montage(frames, cols=6, thumb_w=160, thumb_h=120):
    """
    Builds a grid montage of thumbnails sampled across the sequence
    so the reviewer can see the whole motion, not just the last frame.
    """
    if not frames:
        return None
    n = len(frames)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = cv2.resize(f, (thumb_w, thumb_h))
        canvas[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb
        cv2.putText(canvas, str(i + 1), (c * thumb_w + 4, r * thumb_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


# =====================================================================
# 2. AUTOMATIC CAMERA HARDWARE SCANNER & SELECTOR
# =====================================================================
print("--- PRODUCTION ISL DATA ENGINE (HANDS + FACE) ---")
print("[CAMERA SCAN] Evaluating active video ingestion engines on your host machine...")
available_cameras = []

# Scan the first 5 possible indices to catch integrated and virtual phone sources
for index in range(5):
    test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret:
            available_cameras.append(index)
        test_cap.release()

if not available_cameras:
    print("[CRITICAL ERROR] Execution halted: Zero active capture cards/webcams mapped on system.")
    exit()

print(f"[SCAN COMPLETED] Verified live indices found: {available_cameras}")
print("-> Note: Index 0 is typically the internal laptop webcam.")
print("-> Note: Your phone virtual feed (Smart Connect/DroidCam) will map to Index 1, 2, or 3.")

try:
    selected_index = int(input(f"Enter the index you wish to activate {available_cameras}: ").strip())
    if selected_index not in available_cameras:
        print("[WARNING] Input target was unverified by hardware probe, pursuing link initialization anyway...")
except ValueError:
    selected_index = available_cameras[0]
    print(f"[INFO] Invalid structural parse. Defaulting to lowest index parameter: {selected_index}")

cap = cv2.VideoCapture(selected_index)
if not cap.isOpened():
    print(f"[CRITICAL ERROR] Failed to anchor video stream to index parameter [{selected_index}].")
    exit()

# =====================================================================
# 3. BATCH LABELS WORKFLOW INITIALIZATION
# =====================================================================
mode = input("Select collection mode ('static' or 'dynamic'): ").strip().lower()
if mode not in ['static', 'dynamic']:
    print("[CRITICAL ERROR] Invalid mode selection. Exiting.")
    cap.release()
    exit()

labels_input = input("Enter labels separated by commas (e.g., A,B,C or Hello,Thanks): ").strip()
labels_to_process = [l.strip() for l in labels_input.split(',') if l.strip()]

signer_id = input("Enter signer ID (e.g., signer_1): ").strip() or "signer_unknown"

print("\nWebcam online. Starting multi-label batch collection. Press 'q' inside the window to exit completely.\n")
time.sleep(1)

# =====================================================================
# 4. MULTI-LABEL EXECUTION LOOP
# =====================================================================
for current_label in labels_to_process:
    print(f"\n=========================================\n[ACTIVE BATCH] Now capturing for: '{current_label}' (signer: {signer_id})\n=========================================")

    # Signer-aware folder structure: data/{mode}/{signer_id}/{label}/
    base_dir = os.path.join('data', mode, signer_id, current_label)
    os.makedirs(base_dir, exist_ok=True)

    total_loops = STATIC_TARGET_SAMPLES if mode == 'static' else DYNAMIC_TARGET_SEQUENCES

    existing_indices = [int(os.path.splitext(f)[0]) for f in os.listdir(base_dir)
                         if f.endswith('.npy') and os.path.splitext(f)[0].isdigit()]
    start_idx = (max(existing_indices) + 1) if existing_indices else 0
    target_end_idx = start_idx + total_loops

    sample_num = start_idx
    while sample_num < target_end_idx:
        print(f"\n[SAMPLE PROGRESS] Processing sample {sample_num + 1} (target count {sample_num - start_idx + 1}/{total_loops}) for '{current_label}'")

        # --- PRE-RECORDING LEAD-IN COUNTDOWN ---
        lead_in_start = time.time()
        lead_in_duration = 2.0

        while time.time() - lead_in_start < lead_in_duration:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            time_remaining = max(0.0, lead_in_duration - (time.time() - lead_in_start))
            cv2.putText(frame, f"LABEL: {current_label} | SAMPLE: {sample_num}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"GET READY (Lead-in): {time_remaining:.1f}s", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow('ISL Production Engine Viewport', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Execution killed by user request.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # --- DATA CAPTURE SUB-ROUTINE ---
        capture_success = False
        temp_data_store = None
        preview_frame = None

        if mode == 'static':
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                network_hands_ok = has_valid_hands(results)
                network_face_ok = has_valid_face(results) if REQUIRE_FACE_FOR_STATIC else True

                if network_hands_ok and network_face_ok:
                    temp_data_store = extract_keypoints(results)
                    draw_styled_landmarks(frame, results)
                    preview_frame = frame.copy()
                    capture_success = True
                else:
                    reason = []
                    if not network_hands_ok:
                        reason.append("no hands detected")
                    if not network_face_ok:
                        reason.append("no face detected")
                    print(f"[VALIDATION WARNING] {' and '.join(reason)}. Skipping file serialization. Retrying sample...")
                    cv2.putText(frame, "DETECTION FAILED - RETRYING SAME SAMPLE", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('ISL Production Engine Viewport', frame)
                    cv2.waitKey(1000)
                    continue

        elif mode == 'dynamic':
            sequence_buffer = []
            preview_frames = []
            valid_frame_count = 0
            sequence_start_time = time.time()

            print(f"[LOG] >>> RECORDING RUNNING NOW for '{current_label}' <<<")

            for frame_idx in range(SEQUENCE_LENGTH):
                start_frame_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                draw_styled_landmarks(frame, results)
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)
                preview_frames.append(frame.copy())

                if has_valid_hands(results):
                    valid_frame_count += 1

                cv2.putText(frame, f"RECORDING SEQUENTIAL FLOW | FRAME {frame_idx + 1}/30", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('ISL Production Engine Viewport', frame)
                cv2.waitKey(1)

                elapsed = time.time() - start_frame_time
                if elapsed < FRAME_DELAY:
                    time.sleep(FRAME_DELAY - elapsed)

            actual_duration = time.time() - sequence_start_time
            expected_duration = SEQUENCE_LENGTH * FRAME_DELAY
            if actual_duration > expected_duration * 1.25:
                print(f"[TIMING WARNING] Sequence ran {actual_duration:.2f}s (expected ~{expected_duration:.2f}s). "
                      f"This machine may be too slow for real-time capture — consider flagging this "
                      f"sample set for review or normalizing frame timing in post-processing.")

            temp_data_store = np.array(sequence_buffer)  # Shape: (30, FEATURE_VECTOR_LENGTH)

            valid_ratio = valid_frame_count / SEQUENCE_LENGTH if SEQUENCE_LENGTH else 0
            if len(sequence_buffer) == SEQUENCE_LENGTH and valid_ratio >= MIN_VALID_FRAME_RATIO:
                preview_frame = build_sequence_montage(preview_frames)
                capture_success = True
            else:
                print(f"[VALIDATION WARNING] Sequence rejected — hand detected in only "
                      f"{valid_frame_count}/{SEQUENCE_LENGTH} frames ({valid_ratio:.0%}, need ≥{MIN_VALID_FRAME_RATIO:.0%}). "
                      f"Resetting loop sample step...")
                continue

        # --- AUDIT REVIEW & FILE SYSTEM WRITE GATE ---
        if capture_success:
            decision_made = False
            while not decision_made:
                display_frame = preview_frame.copy()
                bar_y = display_frame.shape[0] - 50
                cv2.rectangle(display_frame, (10, bar_y), (display_frame.shape[1] - 10, display_frame.shape[0] - 10), (0, 0, 0), -1)
                cv2.putText(display_frame, "[SPACE] Accept & Save  |  [R] Reject & Retake", (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('ISL Production Engine Viewport', display_frame)

                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):
                    file_path = os.path.join(base_dir, f"{sample_num}.npy")
                    np.save(file_path, temp_data_store)
                    print(f"[FILE SYSTEM] Successfully committed sample {sample_num} to disk at {file_path}")
                    sample_num += 1
                    decision_made = True
                elif key in (ord('r'), ord('R')):
                    print("[AUDIT REJECTION] Sample explicitly dropped by user feedback loop. Re-running sample pipeline.")
                    decision_made = True
                elif key == ord('q'):
                    print("Execution killed inside confirmation loop.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    print(f"[SUCCESS] Finished target extraction quota for batch class: '{current_label}'")

print("\n=========================================\n[BATCH COMPLETE] All labeled structures saved securely.\n=========================================")
cap.release()
cv2.destroyAllWindows()