"""
Linear Algebra in Python — shared notebook support package.

Importing ``ecp`` gives notebooks a small, stable surface::

    from ecp import header, use_style, animate, validate, draw, linalg

Everything that defines the *look and feel* and the *correctness discipline* of
the course lives here, so the whole course can be restyled or re-checked by
editing one package rather than dozens of notebooks.

The package name is shared with the author's other notebook courses
(*Elementary Computational Physics*, *Molecular and Materials Modelling*) on
purpose: one visual identity, one validation API, one animation player, one
collision-free diagram engine. The course-specific domain helper here is
:mod:`ecp.linalg`, which plays the role ``ecp.mechanics`` plays there.
"""
from . import animate, draw, linalg, validate
from .style import AUTHOR, SERIES_TITLE, SERIES_VERSION, footer, header, use_style

__all__ = [
    "header",
    "footer",
    "use_style",
    "animate",
    "draw",
    "linalg",
    "validate",
    "SERIES_TITLE",
    "SERIES_VERSION",
    "AUTHOR",
]

__version__ = SERIES_VERSION
