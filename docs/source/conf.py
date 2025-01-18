# from setuptools_scm import get_version
# from pathlib import Path
# import importlib

# Generate the version file
# version = get_version()
# version_file = Path('physics/_version.py')
# version_file.write_text(f"__version__ = '{version}'\n")

# Dynamically import the version
# importlib.invalidate_caches()
# physics_module = importlib.import_module("physics")
# __version__ = physics_module.__version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UBC Solar Physics'
copyright = '2024, UBC Solar'
author = 'Joshua Riefman'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # For Google/NumPy style docstrings
    'sphinx.ext.autosummary',  # To generate summary tables for modules
    'myst_parser',           # For Markdown support if needed
]

html_theme = "pydata_sphinx_theme"

autodoc_mock_imports = ['core']
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
