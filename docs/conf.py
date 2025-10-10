# Configuration file for the Sphinx documentation builder.
#
# For full configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# -- Project information -----------------------------------------------------
project = "AriAstro"
copyright = "2025, Varghese Reji"
author = "Varghese Reji"

# -- General configuration ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# List of modules to mock so autodoc won't fail when building docs
autodoc_mock_imports = [
    "astroscrappy",
    "scipy",
    "scipy.ndimage",
    "numpy",
    "astropy"
]

extensions = [
    "sphinx.ext.autodoc",     # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",    # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",    # Add links to highlighted source code
    "sphinxcontrib.mermaid",  # Mermaid diagrams support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"  # Or "sphinx_rtd_theme"

html_logo = "_static/AriAstro_logo.png"
html_static_path = ["_static"]

# Optional: specify Mermaid version
mermaid_version = "10.4.0"
