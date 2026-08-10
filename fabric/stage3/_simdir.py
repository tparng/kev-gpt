"""Portable scratch-dir base for the gate harnesses.

The harnesses were authored on the Windows dev box and defaulted their iverilog
build/scratch dirs to ``C:\\kevbuild`` (outside OneDrive, per the build gotcha).
That path does not exist off that box, so reproduction on Linux/macOS broke.

``kevbuild(*parts)`` returns a build path under, in order of preference:
  1. ``$KEV_SIM_DIR`` if set (point it anywhere you like), else
  2. ``<system-temp>/kevbuild`` (e.g. ``/tmp/kevbuild`` on Linux).

Harnesses that already take a ``--dir`` argument keep it; this is only the
default when none is passed.
"""
from __future__ import annotations

import os
import tempfile


def kevbuild(*parts: str) -> str:
    base = os.environ.get("KEV_SIM_DIR") or os.path.join(tempfile.gettempdir(), "kevbuild")
    return os.path.join(base, *parts)
