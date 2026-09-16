"""Sphinx configuration for Takahe's documentation.

See https://www.sphinx-doc.org/en/master/usage/configuration.html for a
full list of the options available here.
"""

import os
import re
import sys

# -- Path setup ---------------------------------------------------------
#
# Takahe itself lives one directory up from this file (`docs/conf.py` is
# at <repo>/docs/conf.py, and the importable package is at
# <repo>/takahe/), so we add the repository root - not this directory -
# to sys.path. Without this, autodoc cannot import takahe and every
# automodule directive in source/takahe.rst silently produces an empty
# page.
#
# This is resolved relative to __file__ (not the process's current
# working directory) so it works whether Sphinx is invoked from
# `docs/` (as the bundled Makefile does) or from the repository root
# (as Read the Docs does).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 '..')))

import takahe  # noqa: E402  (import after sys.path manipulation, above)

# -- Project information -------------------------------------------------

project = 'Takahe'
copyright = '2026, Sean Richards'
author = 'Sean Richards'

# The full version, including alpha/beta/rc tags. Pulled directly from
# takahe._metadata so the docs never drift from the package version.
release = takahe.__version__
version = release

# -- General configuration ------------------------------------------------

extensions = [
    'recommonmark',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
]

autosectionlabel_prefix_document = True

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'source/modules.rst']

# Both .rst and .md files are treated as documentation sources (recommonmark
# handles the latter).
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# -- Napoleon (docstring -> reST) settings --------------------------------
#
# Takahe's docstrings are Google-style, with one house convention Napoleon
# doesn't understand out of the box: types are written in curly braces and
# separated from the description with a dash/double-dash, e.g.
#
#     Arguments:
#         a {float} -- The semimajor axis, in solar radii.
#
# rather than Napoleon's expected `a (float): description`. The
# fix_docstrings() hook below rewrites recognised lines into the syntax
# Napoleon expects *before* Napoleon parses them, so the API reference
# renders proper parameter/return/raises tables without touching a single
# docstring in the source. See the docstring on fix_docstrings() for the
# exact patterns it handles.

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False
napoleon_include_init_with_doc = True

# -- Autodoc settings ------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Matches a line such as:
#     a {float} -- The semimajor axis, in solar radii.
#     weight {float} -- the BPASS weight (default: {1})
_ARG_RE = re.compile(r'^(\s*)([\w\*]+)\s*\{([^}]*)\}\s*--\s*(.*)$')

# Matches a Raises entry such as:
#     AssertionError -- on malformed input.
#     takahe.TakaheWarning    -- A warning type if ...
_RAISES_RE = re.compile(r'^(\s*)([\w.]+(?:\s*,\s*[\w.]+)*)\s+--\s+(.*)$')

# Matches a Returns/Yields entry such as:
#     {tuple} -- The (m, M) arrays, coerced to baryonic masses.
_RETURN_RE = re.compile(r'^(\s*)\{([^}]*)\}\s*--\s*(.*)$')

# A handful of older docstrings (takahe/__init__.py's integrate_eoms and
# integrate_timescale) use a "Params:" header with bare `name - description`
# entries instead of "Arguments:". Recognise both.
_PARAMS_DASH_RE = re.compile(r'^(\s*)(\w+)\s+-\s+(.*)$')

_SECTION_HEADERS = {
    'arguments:', 'keyword arguments:', 'returns:', 'raises:', 'yields:',
    'params:',
}


def fix_docstrings(app, what, name, obj, options, lines):
    """Rewrites Takahe's house docstring style into Napoleon-compatible
    Google style, in place, before Napoleon processes it.

    This is a Sphinx ``autodoc-process-docstring`` handler - it never
    touches any file on disk, it only transforms the text Sphinx has
    already read out of a docstring, in memory, immediately before
    Napoleon converts that docstring into reST. Numpydoc-style sections
    (used at the bottom of takahe/histogram.py) don't match any of the
    patterns here and pass through untouched.
    """
    section = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        if stripped in _SECTION_HEADERS:
            section = stripped.rstrip(':')
            if section == 'params':
                lines[i] = line.replace('Params', 'Args').replace(
                    'params', 'args')
            continue

        if not line.strip():
            continue

        if line and not line[0].isspace() and section is not None:
            # Dedented back to body text; the section has ended.
            section = None

        if section in ('arguments', 'keyword arguments'):
            m = _ARG_RE.match(line)
            if m:
                indent, pname, ptype, desc = m.groups()
                lines[i] = f"{indent}{pname} ({ptype}): {desc}"
        elif section in ('returns', 'yields'):
            m = _RETURN_RE.match(line)
            if m:
                indent, ptype, desc = m.groups()
                lines[i] = f"{indent}{ptype}: {desc}"
        elif section == 'raises':
            m = _RAISES_RE.match(line)
            if m:
                indent, ename, desc = m.groups()
                lines[i] = f"{indent}{ename}: {desc}"
        elif section == 'params':
            m = _PARAMS_DASH_RE.match(line)
            if m:
                indent, pname, desc = m.groups()
                lines[i] = f"{indent}{pname}: {desc}"


def setup(app):
    # priority=400 runs this before Napoleon's own (default-priority)
    # autodoc-process-docstring handler, so Napoleon sees already-fixed
    # text.
    app.connect('autodoc-process-docstring', fix_docstrings, priority=400)


# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/TakaheLogo.png'
html_theme_options = {
    'logo_only': False,
    'style_external_links': True,
}
