"""Filesystem crawler helpers.

Provides `get_index(root_dir_path: str)` which returns a flat list of
relative file paths (as dicts) under `root_dir_path`.

Behavior:
- Walks the directory tree recursively.
- Skips any file or directory whose name starts with a dot (`.`).
- Skips any files under a `__pycache__` directory.
- Skips `tests/` and vendor directories (node_modules, venv, .venv, etc.).
- Filters files by extension: if a file has an extension, it must be
  one of the common text/code extensions listed in `TEXT_EXTENSIONS`.
  Files without an extension are included (treated as text).
"""

import os
from pathlib import Path
from typing import Dict, List

# Directories to exclude (tests and vendor)
EXCLUDED_DIRS = {
    "tests",
    "test",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "vendor",
    ".tox",
    ".pytest_cache",
    "build",
    "dist",
    "*.egg-info",
    ".mypy_cache",
}

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".pyi",
    ".html",
    ".htm",
    ".css",
    ".js",
}


def get_index(root_dir_path: str) -> List[Dict[str, str]]:
    """Return a flat list of dicts with key `file_path` for files under root.

    `file_path` values are POSIX-style relative paths (strings) relative
    to `root_dir_path`.
    """
    root = Path(root_dir_path).resolve()
    results: List[Dict[str, str]] = []
    if not root.exists() or not root.is_dir():
        return results

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip directories that start with a dot, __pycache__, tests, and vendor dirs
        # Modify dirnames in-place so os.walk will skip them recursively.
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in EXCLUDED_DIRS and d != '__pycache__']

        for fname in filenames:
            if fname.startswith('.'):
                continue

            full = Path(dirpath) / fname

            # Skip files under any __pycache__ (defence in depth)
            if "__pycache__" in full.parts:
                continue

            # Extension based filtering: allow if no extension or ext in TEXT_EXTENSIONS
            ext = full.suffix.lower()
            if ext and ext not in TEXT_EXTENSIONS:
                continue

            # Compute relative POSIX path
            rel = full.relative_to(root).as_posix()
            results.append({"file_path": rel})

    return results


__all__ = ["get_index"]
