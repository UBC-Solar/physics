# These tests are required to satisfy STG Specification #2: Physics Publishing

import importlib


def test_version():
    version_module = importlib.import_module("physics")
    assert isinstance(version_module.__version__, str)
