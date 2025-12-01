# ui/__init__.py
"""
UI package.

app.py imports render_workspace directly from ui.layout.
"""

from .layout import render_workspace

__all__ = ["render_workspace"]
