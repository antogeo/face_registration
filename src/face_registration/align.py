"""Face alignment: warp one face so its landmarks line up with another face's.

We use the 5 landmarks from the detector (eyes, nose, mouth corners) and fit a
similarity transform: rotation + uniform scale + translation. No shearing or
stretching, so the face keeps its shape and only moves into position.
"""

import cv2
import numpy as np


def estimate_alignment(
    src_landmarks: list[tuple[int, int]],
    dst_landmarks: list[tuple[int, int]],
) -> np.ndarray:
    """Return the 2x3 matrix that maps src_landmarks onto dst_landmarks.

    method=LMEDS minimizes the median reprojection error, so unlike the default
    RANSAC it has no pixel threshold to tune and works at any image size.
    """
    src = np.array(src_landmarks, dtype=np.float32)
    dst = np.array(dst_landmarks, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:  # OpenCV returns None when it cannot find a fit
        raise ValueError("Could not estimate an alignment from these landmarks")
    return matrix


def align_face(
    src_img: np.ndarray,
    src_landmarks: list[tuple[int, int]],
    dst_landmarks: list[tuple[int, int]],
    dst_size: tuple[int, int],
) -> np.ndarray:
    """Warp src_img so its face lands on the destination face.

    Returns a new image (numpy array, uint8) of shape (height, width, channels)
    where (width, height) = dst_size. It is src_img moved by the matrix from
    estimate_alignment, so src_landmarks end up on top of dst_landmarks.

    - Same size as the destination image, so the two can be overlaid pixel
      for pixel (e.g. for blending).
    - Pixels that come from outside src_img are black.
    - Colours are unchanged; only positions move (bilinear interpolation).
    - src_img itself is not modified.
    """
    matrix = estimate_alignment(src_landmarks, dst_landmarks)
    return cv2.warpAffine(src_img, matrix, dst_size)
