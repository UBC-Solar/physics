import importlib


def test_version():
    version_module = importlib.import_module("physics.__version__")
    assert isinstance(version_module.__version__, str)
