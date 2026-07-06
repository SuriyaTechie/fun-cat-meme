import os

import cv2

FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"


def _candidate_paths():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(cv2.data.haarcascades, FACE_CASCADE_FILE),
        os.path.join(project_dir, "assets", "haarcascades", FACE_CASCADE_FILE),
        os.path.join(project_dir, FACE_CASCADE_FILE),
    ]


def resolve_face_cascade_path():
    for candidate in _candidate_paths():
        if os.path.exists(candidate):
            return candidate

    searched_paths = "\n".join(f"- {path}" for path in _candidate_paths())
    raise FileNotFoundError(
        "Could not find OpenCV's face cascade XML file.\n"
        "Searched these paths:\n"
        f"{searched_paths}\n"
        "Place 'haarcascade_frontalface_default.xml' in assets/haarcascades/ "
        "or reinstall OpenCV with the data files."
    )


def load_face_cascade():
    cascade_path = resolve_face_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from '{cascade_path}'.")

    return face_cascade, cascade_path
