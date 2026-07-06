import os

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

HAND_LANDMARKER_MODEL_FILE = "hand_landmarker.task"


def _candidate_paths():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(project_dir, "assets", "mediapipe", HAND_LANDMARKER_MODEL_FILE),
        os.path.join(project_dir, HAND_LANDMARKER_MODEL_FILE),
    ]


def resolve_hand_landmarker_model_path():
    for candidate in _candidate_paths():
        if os.path.exists(candidate):
            return candidate

    searched_paths = "\n".join(f"- {path}" for path in _candidate_paths())
    raise FileNotFoundError(
        "Could not find the MediaPipe hand landmarker model file.\n"
        "Searched these paths:\n"
        f"{searched_paths}\n"
        "Place 'hand_landmarker.task' in assets/mediapipe/."
    )


def load_hand_landmarker():
    model_path = resolve_hand_landmarker_model_path()
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = hand_landmarker.HandLandmarker.create_from_options(options)
    return detector, model_path


def create_mp_image_from_bgr(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))


def draw_hand_landmarks(frame, landmarks):
    if not landmarks:
        return

    height, width = frame.shape[:2]
    pixel_points = []
    for landmark in landmarks:
        x_px = min(int(landmark.x * width), width - 1)
        y_px = min(int(landmark.y * height), height - 1)
        pixel_points.append((x_px, y_px))

    for connection in hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS:
        start_point = pixel_points[connection.start]
        end_point = pixel_points[connection.end]
        cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    for point in pixel_points:
        cv2.circle(frame, point, 4, (255, 255, 255), -1)
        cv2.circle(frame, point, 6, (0, 128, 255), 1)
