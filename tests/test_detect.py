import numpy as np

from face_registration.detect import crop_face, face_center


def test_face_center_returns_middle_of_box():
    # Box starts at (10, 20), is 100 wide and 50 tall.
    # Center x = 10 + 100 // 2 = 60, center y = 20 + 50 // 2 = 45.
    assert face_center((10, 20, 100, 50)) == (60, 45)


def test_crop_face_adds_margin_around_box():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Box is 40x40, margin 0.2 -> 8 px extra on each side -> 40 + 8 + 8 = 56.
    crop = crop_face(img, (50, 50, 40, 40), margin=0.2)
    assert crop.shape == (56, 56, 3)  # (rows=height, cols=width, channels)


def test_crop_face_is_clipped_at_image_border():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Top-left corner: the margin would go to -8, but is clipped to 0 -> 0..48.
    assert crop_face(img, (0, 0, 40, 40), margin=0.2).shape == (48, 48, 3)
    # Bottom-right corner: box ends at 200, margin would reach 206, clipped to 200.
    # Width/height: 164..200 = 36.
    assert crop_face(img, (170, 170, 30, 30), margin=0.2).shape == (36, 36, 3)


def test_crop_face_returns_a_copy_not_a_view():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(img, (20, 20, 40, 40), margin=0.0)
    crop[:] = 255  # paint the crop white
    assert img.max() == 0  # the original image must be untouched
