"""Utility helpers for the project (logging helpers)."""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the application.

    The logger prints to stderr with a simple format. Multiple calls
    return the same logger instance (handlers are added only once).
    """
    logger_name = name or "ghsummarizer"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yml. If None, looks for config.yml in project root.
        
    Returns:
        Dictionary containing configuration settings.
    """
    if config_path is None:
        # Default to config.yml in project root (parent of src/)
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.yml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "skipped_dirs": ["example", "examples", "test", "tests", "contrib"],
            "skipped_patterns": [],
            "class_methods": False,
            "max_prompt_tokens": 6000
        }
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {
            "skipped_dirs": ["example", "examples", "test", "tests", "contrib"],
            "skipped_patterns": [],
            "class_methods": False,
            "max_prompt_tokens": 6000
        }


def get_skipped_dirs(config_path: Optional[Path] = None) -> list[str]:
    """Get list of directory names to skip during skeleton generation.
    
    Args:
        config_path: Path to config.yml. If None, uses default location.
        
    Returns:
        List of directory name patterns to skip (case-insensitive).
    """
    config = load_config(config_path)
    return config.get("skipped_dirs", [])


def get_class_methods_flag(config_path: Optional[Path] = None) -> bool:
    """Get whether to include class methods in skeleton output.
    
    Args:
        config_path: Path to config.yml. If None, uses default location.
        
    Returns:
        True if class methods should be included, False to show only class names.
    """
    config = load_config(config_path)
    return config.get("class_methods", False)


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


__all__ = ["get_logger", "clone_repo", "load_config", "get_skipped_dirs", "get_class_methods_flag"]
