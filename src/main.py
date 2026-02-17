"""Small async runner that clones the `psf/requests` repo using `src.cloner.clone_repo`.

Usage:
  - Ensure `DATA_DIR` is set and writable, e.g.:
      export DATA_DIR="$PWD/.data"
      mkdir -p "$DATA_DIR"
  - Run as a module from repository root:
      python -m src.main

This script prints the clone path and removes the clone when finished.
"""

import asyncio

from src.cloner import clone_repo
from src.crawler.index import get_index
from src.crawler.naive_skeleton import code_skeleton

REPO_URL = "https://github.com/psf/requests"


async def run() -> int:
    try:
        path = await clone_repo(REPO_URL, timeout=180)
    except Exception as exc:
        print(f"Clone failed: {exc}")
        return 1

    print(f"Cloned repository to: {path}")
    # Basic sanity check: ensure README or setup.py exists
    if any((path / name).exists() for name in ("README.md", "setup.py", "pyproject.toml")):
        print("Basic repo files detected.")
    else:
        print("Warning: common top-level files not detected.")

    # Keep the cloned repository on disk for inspection. Do not remove it.
    print(f"Clone retained at: {path}")

    # Index the repository files and show a brief summary
    try:
        indexed = get_index(str(path))
        print(f"Indexed {len(indexed)} files.")
        if indexed:
            sample = indexed[:10]
            print("Sample files:")
            for entry in sample:
                print(" -", entry.get("file_path"))
    except Exception as e:
        print(f"Indexing failed: {e}")

    # Generate a naive code skeleton from the index and save it into the repo
    try:
        skeleton_text = code_skeleton(indexed, root_dir=str(path))
        # Print a short preview (first 40 lines)
        preview = "\n".join(skeleton_text.splitlines()[:40])
        print("\n=== Skeleton preview ===\n")
        print(preview)

        out_file = path / "SKELETON.txt"
        out_file.write_text(skeleton_text, encoding="utf-8")
        print(f"Full skeleton written to: {out_file}")
    except Exception as e:
        print(f"Skeleton generation failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
