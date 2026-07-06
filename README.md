# Gesture Meme Reactor

## Install

Install the required packages with:

```bash
pip install opencv-contrib-python mediapipe numpy
```

> Use `opencv-contrib-python` so the built-in `cv2.face` module is available.

The project also includes a fallback face-cascade XML in `assets/haarcascades/`
because some OpenCV wheels expose `cv2.data.haarcascades` without shipping the
actual `haarcascade_frontalface_default.xml` file.

The project also includes a local MediaPipe hand model in `assets/mediapipe/`
because this environment's `mediapipe` package uses the Tasks API and expects a
`hand_landmarker.task` asset at runtime.

The project also includes a local MediaPipe face model in `assets/mediapipe/`
for face-landmark-based features such as tongue-out detection.

## Run order

1. `python enroll_face.py`
2. `python train_recognizer.py`
3. `python upload_memes.py` to pick cat meme photos from your gallery/file picker (optional)
4. `python gesture_meme_reactor.py`

## What each script does

- `enroll_face.py`: opens the webcam, detects a face with Haar cascade, and saves 50 grayscale face crops to `dataset/suriya/`.
- `train_recognizer.py`: reads all face images from `dataset/`, trains an LBPH model, and writes `trainer.yml` plus `labels.json`.
- `upload_memes.py`: opens a file picker so you can choose one cat meme photo for all gestures or separate photos for each gesture, including `Tongue Out`.
- `gesture_meme_reactor.py`: opens the webcam, optionally checks for a verified recognized face, detects a hand gesture with MediaPipe, detects a tongue-out face expression, and displays a matching meme image (or an auto-generated placeholder when the meme image is missing).

## How it works

- `enroll_face.py` saves one face image every third successful face detection.
- `train_recognizer.py` maps each folder name in `dataset/` to a numeric label and trains OpenCV LBPH.
- `gesture_meme_reactor.py` uses `cv2.face` for face verification, MediaPipe Hand Landmarker for gestures, and MediaPipe Face Landmarker plus a mouth-color heuristic for the `Tongue Out` trigger.

## Face confidence threshold

The face model uses a confidence threshold of `70`. Lower this number to require a stronger match, or raise it to accept more faces.

Example tuning:

```python
FACE_CONFIDENCE_THRESHOLD = 60.0
```

A lower threshold is stricter and will only verify closer matches. A higher threshold is more lenient and may accept more faces.
