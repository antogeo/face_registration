"""Face detection: load an image, find faces, compute their center, and crop them.

Uses OpenCV's YuNet detector (cv2.FaceDetectorYN) with the 2023mar model.
That model has a FIXED input size, so every image is resized and padded to
MODEL_INPUT_SIZE before detection, and the results are scaled back to the
original image. See models/README.md for how to download the model.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# A face box is (x, y, w, h): top-left corner plus width and height, in pixels.
Box = tuple[int, int, int, int]

# Repo root is two folders up from this file: src/face_registration/detect.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"

# The (width, height) the 2023mar model was exported with. It only accepts this size.
MODEL_INPUT_SIZE = (640, 640)


@dataclass
class Face:
    box: Box
    # 5 (x, y) points: right eye, left eye, nose tip, right mouth corner, left mouth corner.
    # "Right" means the person's right, which is on the left side of the image.
    landmarks: list[tuple[int, int]]
    score: float  # detector confidence, 0..1


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR array. Raises if the file can't be read."""
    img = cv2.imread(str(path))
    if img is None:  # cv2.imread does not raise on failure, it returns None
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize img to fit inside size (w, h) without stretching, pad the rest with black.

    Returns the padded image and the scale factor used (new / original).
    The image sits in the top-left corner, so mapping back is just dividing by scale.
    """
    target_w, target_h = size
    img_h, img_w = img.shape[:2]
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = round(img_w * scale), round(img_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def detect_faces(
    img: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = 0.9,
) -> list[Face]:
    """Return the faces found in the image, largest first. Coordinates are in the original image."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found: {model_path} (see models/README.md)")

    model_input, scale = _letterbox(img, MODEL_INPUT_SIZE)
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",  # config file, not needed for ONNX models
        MODEL_INPUT_SIZE,  # must match the model's fixed input size
        score_threshold=score_threshold,  # lower = more faces found, but more false ones
    )
    _, rows = detector.detect(model_input)
    if rows is None:  # YuNet returns None (not an empty array) when it finds nothing
        return []

    # Each row has 15 numbers: x, y, w, h, then 5 landmark (x, y) pairs, then the score.
    # Dividing by scale converts from model-input pixels back to original-image pixels.
    faces = []
    for r in rows:
        box = tuple(round(float(v) / scale) for v in r[:4])
        landmarks = [(round(r[i] / scale), round(r[i + 1] / scale)) for i in range(4, 14, 2)]
        faces.append(Face(box=box, landmarks=landmarks, score=float(r[14])))
    # could also be written as:
    # def box_area(f):
    #     return f.box[2] * f.box[3]

    # return sorted(faces, key=box_area, reverse=True)    
    return sorted(faces, key=lambda f: f.box[2] * f.box[3], reverse=True)


def face_center(box: Box) -> tuple[int, int]:
    """Return the (x, y) center point of a face box."""
    x, y, w, h = box
    return x + w // 2, y + h // 2


def crop_face(img: np.ndarray, box: Box, margin: float = 0.2) -> np.ndarray:
    """Crop a face from the image, with extra margin around the box.

    margin=0.2 adds 20% of the box size on each side. The crop is clipped
    so it never goes outside the image.
    """
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    img_h, img_w = img.shape[:2]
    x0, y0 = max(x - dx, 0), max(y - dy, 0)
    x1, y1 = min(x + w + dx, img_w), min(y + h + dy, img_h)
    return img[y0:y1, x0:x1].copy()
