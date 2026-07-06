import os

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV is not installed in the active environment. "
        "Install it with: pip install opencv-contrib-python"
    ) from exc

from face_cascade_utils import load_face_cascade

# Save collected face images into this folder.
DATASET_DIR = os.path.join("dataset", "suriya")
os.makedirs(DATASET_DIR, exist_ok=True)

face_cascade, cascade_path = load_face_cascade()
print(f"Using face cascade: {cascade_path}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open the webcam. Make sure your camera is connected.")

captured_count = 0
frame_index = 0

print("Starting enrollment. Press q to quit early.")
while cap.isOpened() and captured_count < 50:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read a frame from the webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if frame_index % 3 == 0 and len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (200, 200))
        image_path = os.path.join(DATASET_DIR, f"{captured_count}.jpg")
        cv2.imwrite(image_path, face)
        captured_count += 1
        print(f"Captured {captured_count}/50")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Captured: {captured_count}/50",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Enroll Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print("Enrollment finished.")
