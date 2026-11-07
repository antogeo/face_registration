import numpy as np

from face_registration.align import align_face, estimate_alignment

LANDMARKS = [(30, 40), (70, 40), (50, 60), (35, 80), (65, 80)]


def test_identical_landmarks_give_identity_transform():
    matrix = estimate_alignment(LANDMARKS, LANDMARKS)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    assert np.allclose(matrix, identity, atol=1e-3)


def test_pure_shift_is_recovered():
    shifted = [(x + 10, y + 5) for x, y in LANDMARKS]
    matrix = estimate_alignment(LANDMARKS, shifted)
    # Rotation/scale part stays identity, translation part is (10, 5).
    assert np.allclose(matrix[:, :2], np.eye(2), atol=1e-3)
    assert np.allclose(matrix[:, 2], [10, 5], atol=1e-2)


def test_scale_is_recovered():
    doubled = [(x * 2, y * 2) for x, y in LANDMARKS]
    matrix = estimate_alignment(LANDMARKS, doubled)
    assert np.allclose(matrix[:, :2], 2 * np.eye(2), atol=1e-3)


def test_align_face_moves_a_pixel_to_the_matching_position():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40, 30] = 255  # white pixel on the first landmark (x=30, y=40)
    shifted = [(x + 10, y + 5) for x, y in LANDMARKS]
    out = align_face(img, LANDMARKS, shifted, dst_size=(100, 100))
    assert out[45, 40].max() == 255  # now at (x=40, y=45)
    assert out.shape == (100, 100, 3)
