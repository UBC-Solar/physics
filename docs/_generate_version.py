from setuptools_scm import get_version
from pathlib import Path

# Get the version from setuptools_scm
version = get_version()

# Path to the version file specified in your pyproject.toml
version_file = Path('physics/_version.py')

# Write the version into _version.py
version_file.write_text(f"__version__ = '{version}'\n")
