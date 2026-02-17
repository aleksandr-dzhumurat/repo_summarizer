import asyncio
import os
import shutil
from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)


async def clone_repo(repo_url: str, timeout: int = 120) -> Path:
    """Clone `repo_url` into a subdirectory under `os.environ['DATA_DIR']`.

    The directory is created with `tempfile.mkdtemp(dir=DATA_DIR, prefix=...)`.
    If `DATA_DIR` is not set in the environment a RuntimeError is raised.
    On failure the created directory is removed and a RuntimeError is raised.
    """
    try:
        base_dir = os.environ["DATA_DIR"]
    except KeyError:
        raise RuntimeError("Environment variable DATA_DIR must be set and writable")

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Derive a persistent path for this repo: DATA_DIR/<owner>/<repo>
    slug = repo_url.rstrip("/")
    if "github.com/" in slug:
        slug = slug.split("github.com/")[-1]
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]

    target_path = base_path.joinpath(*slug.split("/"))

    # If repository already cloned, skip cloning and return path
    if target_path.exists():
        logger.info("Repository already present at %s — skipping clone", target_path)
        return target_path

    # Ensure parent exists and perform clone into the target path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth=1",
        "--single-branch",
        "--no-tags",
        repo_url,
        str(target_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        # Remove partially created target
        shutil.rmtree(target_path, ignore_errors=True)
        raise RuntimeError("Clone timed out")

    if proc.returncode != 0:
        # Cleanup on failure
        shutil.rmtree(target_path, ignore_errors=True)
        err = stderr.decode(errors="ignore").strip()
        logger.error("Clone failed for %s: %s", repo_url, err)
        raise RuntimeError(f"Clone failed: {err}")

    logger.info("Successfully cloned %s -> %s", repo_url, target_path)
    return Path(target_path)


__all__ = ["clone_repo"]
