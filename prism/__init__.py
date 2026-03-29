"""Repository-root compatibility shim for the local Prism source tree.

This makes `import prism` from the repo root resolve to `src/prism` without
requiring `PYTHONPATH=src` or a prior editable install.
"""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _ROOT / "src" / "prism"

if not _SRC_PACKAGE.exists():
    raise ImportError(f"Could not locate the local Prism package at {_SRC_PACKAGE}")

# Make submodule imports resolve into the source tree.
__path__ = [str(_SRC_PACKAGE)]
__file__ = str(_SRC_PACKAGE / "__init__.py")

with open(_SRC_PACKAGE / "__init__.py", "r", encoding="utf-8") as handle:
    code = compile(handle.read(), __file__, "exec")
    exec(code, globals())
