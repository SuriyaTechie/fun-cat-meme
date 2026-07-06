import os

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

FACE_LANDMARKER_MODEL_FILE = "face_landmarker.task"
TONGUE_OUT_TRIGGER = "Tongue Out"

UPPER_LIP_CENTER_INDEX = 13
LOWER_LIP_CENTER_INDEX = 14
LEFT_MOUTH_INDEX = 78
RIGHT_MOUTH_INDEX = 308
FACE_TOP_INDEX = 10
FACE_BOTTOM_INDEX = 152
INNER_MOUTH_INDICES = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
]


def _candidate_paths():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(project_dir, "assets", "mediapipe", FACE_LANDMARKER_MODEL_FILE),
        os.path.join(project_dir, FACE_LANDMARKER_MODEL_FILE),
    ]


def resolve_face_landmarker_model_path():
    for candidate in _candidate_paths():
        if os.path.exists(candidate):
            return candidate

    searched_paths = "\n".join(f"- {path}" for path in _candidate_paths())
    raise FileNotFoundError(
        "Could not find the MediaPipe face landmarker model file.\n"
        "Searched these paths:\n"
        f"{searched_paths}\n"
        "Place 'face_landmarker.task' in assets/mediapipe/."
    )


def load_face_landmarker():
    model_path = resolve_face_landmarker_model_path()
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    detector = face_landmarker.FaceLandmarker.create_from_options(options)
    return detector, model_path


def _landmark_to_pixel(landmark, width, height):
    x_px = min(max(int(landmark.x * width), 0), width - 1)
    y_px = min(max(int(landmark.y * height), 0), height - 1)
    return x_px, y_px


def _blendshape_score(categories, target_index):
    for category in categories:
        if category.index == int(target_index):
            return category.score
    return 0.0


def detect_tongue_out(frame, face_result):
    if not face_result.face_landmarks:
        return False

    height, width = frame.shape[:2]
    landmarks = face_result.face_landmarks[0]
    blendshapes = face_result.face_blendshapes[0] if face_result.face_blendshapes else []

    upper_lip = landmarks[UPPER_LIP_CENTER_INDEX]
    lower_lip = landmarks[LOWER_LIP_CENTER_INDEX]
    left_mouth = landmarks[LEFT_MOUTH_INDEX]
    right_mouth = landmarks[RIGHT_MOUTH_INDEX]
    face_top = landmarks[FACE_TOP_INDEX]
    face_bottom = landmarks[FACE_BOTTOM_INDEX]

    mouth_width = max(abs(right_mouth.x - left_mouth.x), 1e-6)
    lip_gap = abs(lower_lip.y - upper_lip.y)
    face_height = max(abs(face_bottom.y - face_top.y), 1e-6)
    jaw_open_score = _blendshape_score(blendshapes, face_landmarker.Blendshapes.JAW_OPEN)

    mouth_open_enough = (
        jaw_open_score >= 0.18
        or lip_gap / mouth_width >= 0.16
        or lip_gap / face_height >= 0.045
    )
    if not mouth_open_enough:
        return False

    mouth_polygon = np.array(
        [_landmark_to_pixel(landmarks[index], width, height) for index in INNER_MOUTH_INDICES],
        dtype=np.int32,
    )
    mouth_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mouth_mask, [mouth_polygon], 255)

    mouth_center_y = _landmark_to_pixel(lower_lip, width, height)[1]
    lower_mask = np.zeros_like(mouth_mask)
    lower_mask[mouth_center_y:, :] = 255
    mouth_mask = cv2.bitwise_and(mouth_mask, lower_mask)

    mouth_area = cv2.countNonZero(mouth_mask)
    if mouth_area < 50:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tongue_mask_1 = cv2.inRange(hsv, (0, 35, 55), (20, 255, 255))
    tongue_mask_2 = cv2.inRange(hsv, (160, 20, 55), (179, 255, 255))
    tongue_mask = cv2.bitwise_or(tongue_mask_1, tongue_mask_2)
    tongue_mask = cv2.bitwise_and(tongue_mask, mouth_mask)

    red_pixels = cv2.countNonZero(tongue_mask)
    red_ratio = red_pixels / float(mouth_area)
    if red_pixels < 25 or red_ratio < 0.12:
        return False

    ys, xs = np.where(tongue_mask > 0)
    if len(ys) == 0:
        return False

    centroid_y = float(np.mean(ys))
    upper_lip_y = _landmark_to_pixel(upper_lip, width, height)[1]
    lower_lip_y = _landmark_to_pixel(lower_lip, width, height)[1]
    if centroid_y <= (upper_lip_y + lower_lip_y) / 2.0:
        return False

    return True
