# Configuration file for the Sphinx documentation builder.

import os
import sys

# Don't try to import the source code during docs build
# This prevents import errors when dependencies aren't available
# sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'AI-Powered Resume & Job Matcher'
copyright = '2024, AI Resume Matcher Team'
author = 'AI Resume Matcher Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.githubpages',
]

# Disable autodoc and other extensions that require importing code
# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.napoleon',
# ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
