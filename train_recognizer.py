import json
import os

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

DATASET_DIR = "dataset"
LABELS_FILE = "labels.json"
TRAINER_FILE = "trainer.yml"

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder '{DATASET_DIR}' not found. Run enroll_face.py first and create at least one label folder."
    )

face_images = []
face_labels = []
label_names = {}

for root, _, files in os.walk(DATASET_DIR):
    label_name = os.path.basename(root)
    if root == DATASET_DIR or not label_name:
        continue

    if label_name not in label_names:
        label_names[label_name] = len(label_names)

    for file_name in sorted(files):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(root, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: could not read image {image_path}")
            continue

        image = cv2.resize(image, (200, 200))
        face_images.append(image)
        face_labels.append(label_names[label_name])

if len(face_images) == 0:
    raise RuntimeError(
        f"No training images found under '{DATASET_DIR}'. "
        "Run enroll_face.py first or add face images to a label folder such as dataset/<name>/."
    )

if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    raise RuntimeError(
        "OpenCV does not expose cv2.face.LBPHFaceRecognizer_create(). Install opencv-contrib-python."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_images, np.array(face_labels, dtype=np.int32))
recognizer.write(TRAINER_FILE)

id_to_name = {str(value): key for key, value in label_names.items()}
with open(LABELS_FILE, "w", encoding="utf-8") as fp:
    json.dump(id_to_name, fp, indent=2)

print(f"Training complete. Model saved to {TRAINER_FILE}")
print(f"Label mapping saved to {LABELS_FILE}")
