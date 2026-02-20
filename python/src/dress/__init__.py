"""
dress - Python bindings for the DRESS (Diffusive Recursive Structural
Similarity) graph library.

Uses the compiled C extension (``dress._core``) when available, otherwise
falls back to the pure-Python implementation (``dress.core``).
"""

try:
    from dress._core import *  # noqa: F401,F403

    _BACKEND = "c"
except ImportError:
    from dress.core import *  # noqa: F401,F403

    _BACKEND = "python"
