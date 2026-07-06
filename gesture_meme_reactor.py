import json
import os
import time

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV is not installed in the active environment. "
        "Install it with: pip install opencv-contrib-python"
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "NumPy is not installed in the active environment. "
        "Install it with: pip install numpy"
    ) from exc

from face_cascade_utils import load_face_cascade
try:
    from face_landmarker_utils import (
        TONGUE_OUT_TRIGGER,
        detect_tongue_out,
        load_face_landmarker,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("mediapipe"):
        raise ModuleNotFoundError(
            "MediaPipe is not installed in the active environment. "
            "Install it with: pip install mediapipe"
        ) from exc
    raise
try:
    from hand_landmarker_utils import (
        create_mp_image_from_bgr,
        draw_hand_landmarks,
        load_hand_landmarker,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("mediapipe"):
        raise ModuleNotFoundError(
            "MediaPipe is not installed in the active environment. "
            "Install it with: pip install mediapipe"
        ) from exc
    raise
from meme_assets import GESTURE_FILES, MEMES_DIR

TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
FACE_CONFIDENCE_THRESHOLD = 70.0


def load_recognizer():
    if os.path.exists(TRAINER_FILE) and os.path.exists(LABELS_FILE):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            print("Warning: cv2.face.LBPHFaceRecognizer_create() is not available.")
            return None, {}

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)
        with open(LABELS_FILE, "r", encoding="utf-8") as fp:
            label_map = json.load(fp)
        return recognizer, label_map

    print("Warning: trainer.yml or labels.json not found. Running gesture-only mode.")
    return None, {}


def make_placeholder(gesture_name, file_name):
    """Create a dark-gray placeholder image for a missing meme file.

    The image is 400x300 and contains two lines of text guiding the user.
    """
    h, w = 300, 400
    img = np.full((h, w, 3), 40, dtype=np.uint8)  # dark gray background
    title = f"No meme yet for: {gesture_name}"
    hint = f"Add your cat photo to memes/{file_name}"

    # Title (larger)
    cv2.putText(img, title, (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    # Hint (smaller)
    cv2.putText(img, hint, (12, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def load_memes():
    """Return a dict mapping gesture name -> image (real or placeholder).

    Always returns an image for every known gesture so the reactor never fails
    due to missing meme files.
    """
    os.makedirs(MEMES_DIR, exist_ok=True)
    memes = {}
    for gesture, filename in GESTURE_FILES.items():
        path = os.path.join(MEMES_DIR, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is None:
                print(f"Info: failed to load {path}; using placeholder")
                img = make_placeholder(gesture, filename)
            else:
                print(f"Info: loaded meme for {gesture} from {path}")
        else:
            print(f"Info: meme not found for {gesture}; creating placeholder")
            img = make_placeholder(gesture, filename)

        # Ensure consistent size (400x300)
        img = cv2.resize(img, (400, 300))
        memes[gesture] = img

    return memes


def load_meme_image(gesture_name):
    file_name = GESTURE_FILES.get(gesture_name)
    if not file_name:
        return None

    meme_path = os.path.join(MEMES_DIR, file_name)
    if not os.path.exists(meme_path):
        print(f"Warning: meme image not found: {meme_path}")
        return None

    return cv2.imread(meme_path)


def finger_states_from_landmarks(landmarks, handedness):
    fingers = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    }

    if handedness not in ("Left", "Right"):
        handedness = "Right"

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness == "Right":
        fingers["thumb"] = thumb_tip.x > thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x < thumb_ip.x

    fingers["index"] = landmarks[8].y < landmarks[6].y
    fingers["middle"] = landmarks[12].y < landmarks[10].y
    fingers["ring"] = landmarks[16].y < landmarks[14].y
    fingers["pinky"] = landmarks[20].y < landmarks[18].y

    return fingers


def classify_gesture(finger_states):
    thumb = finger_states["thumb"]
    index = finger_states["index"]
    middle = finger_states["middle"]
    ring = finger_states["ring"]
    pinky = finger_states["pinky"]

    if thumb and index and middle and ring and pinky:
        return "Open Palm"
    if not thumb and index and middle and ring and pinky:
        return "OK"
    if not thumb and index and middle and not ring and not pinky:
        return "Peace"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up"
    if not thumb and not index and not middle and not ring and not pinky:
        return "Fist"
    return None


def main():
    recognizer, label_map = load_recognizer()

    face_cascade, cascade_path = load_face_cascade()
    print(f"Using face cascade: {cascade_path}")
    face_landmarker_detector, face_model_path = load_face_landmarker()
    print(f"Using face landmarker model: {face_model_path}")
    hand_landmarker_detector, hand_model_path = load_hand_landmarker()
    print(f"Using hand landmarker model: {hand_model_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the webcam. Make sure your camera is connected.")

    memes = load_memes()

    last_trigger = None
    last_switch_time = time.time()
    current_meme = None
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = create_mp_image_from_bgr(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            verified_person = False

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if recognizer is not None and len(faces) > 0:
                x, y, w, h = faces[0]
                face = gray[y : y + h, x : x + w]
                face = cv2.resize(face, (200, 200))
                label, confidence = recognizer.predict(face)
                if confidence < FACE_CONFIDENCE_THRESHOLD:
                    verified_person = True
                    cv2.putText(
                        frame,
                        f"Verified ({confidence:.1f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"Unverified ({confidence:.1f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            else:
                if recognizer is None:
                    # No recognizer available: run in gesture-only mode.
                    verified_person = True
                    cv2.putText(
                        frame,
                        "Gesture-only mode",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

            face_result = face_landmarker_detector.detect_for_video(mp_image, frame_index * 33)
            tongue_out_detected = detect_tongue_out(frame, face_result)

            results = hand_landmarker_detector.detect_for_video(mp_image, frame_index * 33)
            gesture_name = None
            if results.hand_landmarks and results.handedness:
                hand_landmarks = results.hand_landmarks[0]
                handedness = results.handedness[0][0].category_name
                states = finger_states_from_landmarks(hand_landmarks, handedness)
                gesture_name = classify_gesture(states)
                draw_hand_landmarks(frame, hand_landmarks)

                if gesture_name:
                    cv2.putText(
                        frame,
                        gesture_name,
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

            trigger_name = TONGUE_OUT_TRIGGER if tongue_out_detected else gesture_name
            if tongue_out_detected:
                cv2.putText(
                    frame,
                    TONGUE_OUT_TRIGGER,
                    (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )

            if verified_person and trigger_name:
                if trigger_name != last_trigger and time.time() - last_switch_time >= 1.0:
                    last_trigger = trigger_name
                    last_switch_time = time.time()
                    current_meme = memes.get(trigger_name)
            else:
                current_meme = None

            cv2.imshow("Gesture Meme Reactor", frame)
            if current_meme is not None:
                cv2.imshow("Meme", current_meme)
            else:
                blank = 255 * np.ones((300, 400, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No meme to display",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                )
                cv2.imshow("Meme", blank)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1
    finally:
        face_landmarker_detector.close()
        hand_landmarker_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
