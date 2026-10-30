"""Smoke test: ensure the package imports and exposes a version."""

import face_registration


def test_package_imports():
    assert face_registration.__version__


def test_version_is_string():
    assert isinstance(face_registration.__version__, str)
