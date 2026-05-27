"""Path-handling helpers that defend against path traversal.

Filenames from user uploads (`UploadFile.filename`) and any path that
round-trips through the database are treated as untrusted. We:

  * strip the directory component (`os.path.basename`) so `../etc/passwd`
    becomes `passwd`,
  * reject names that still contain path separators or null bytes after
    that step,
  * resolve the final destination and assert it stays under the intended
    root.

The helpers raise `HTTPException(400)` on traversal attempts so the
caller does not need bespoke error handling.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException


def safe_filename(raw: str | None, default: str = "scan") -> str:
    """Return a sanitised, separator-free filename.

    Raises HTTPException(400) when the input contains a null byte, which
    is a clear traversal/log-poisoning signal.
    """
    if raw is None or raw == "":
        return default
    if "\x00" in raw:
        raise HTTPException(status_code=400, detail="Invalid filename")
    # os.path.basename collapses `../foo`, `..\\foo`, `/abs/foo` to `foo`.
    name = os.path.basename(raw)
    # `..` alone, or empty after basename, falls back to the default.
    if name in ("", ".", ".."):
        return default
    # Disallow remaining separators (handles odd encodings on Windows
    # roundtrips and explicit backslashes).
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def safe_db_path(stored: str) -> Path:
    """Resolve a DB-stored path and confirm it lives strictly under ``upload_dir``.

    All scan/prediction artefacts are written under ``settings.upload_dir``
    by trusted server code.  Re-asserting that invariant here defends
    against future bugs that would let a tampered DB row point outside.
    """
    from app.config import settings

    upload_root = settings.upload_dir.resolve()
    try:
        resolved = Path(stored).resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid stored path") from exc
    if upload_root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid stored path")
    return resolved


def safe_join(root: Path, *parts: str) -> Path:
    """Join `parts` onto `root` and ensure the result stays inside `root`.

    Resolves symlinks too — if any segment escapes `root` (`..`, absolute
    path, or symlink pointing outside), raises HTTPException(400).
    """
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts)
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    # `is_relative_to` is 3.9+; we target 3.11 per pyproject.
    if not (resolved == root_resolved or root_resolved in resolved.parents):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved
