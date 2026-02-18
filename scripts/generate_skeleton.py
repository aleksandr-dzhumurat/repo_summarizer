#!/usr/bin/env python3
"""Generate code skeleton for a GitHub repository.

This script:
1. Clones the repository (if not already present)
2. Indexes all files in the repository
3. Generates a code skeleton (SKELETON.txt)

The skeleton includes imports and function signatures from Python files,
which can be used for analysis or as input to LLM summarization.

Usage:
    python3 scripts/generate_skeleton.py [repo_url]
    
Environment variables:
    DATA_DIR: Directory where repositories are cloned (required)
    REPO: Default repository URL if not provided as argument
"""

import asyncio
import os
import sys

from src.crawler.index import get_index
from src.crawler.naive_skeleton import code_skeleton
from src.utils import clone_repo, get_logger

logger = get_logger(__name__)


async def generate_skeleton(repo_url: str) -> int:
    """Clone repository (if needed) and generate code skeleton.
    
    Args:
        repo_url: GitHub repository URL to process
        
    Returns:
        0 on success, 1 on failure
    """
    print(f"Repository: {repo_url}")
    
    # Step 1: Clone repository (or use existing clone)
    try:
        repo_path = await clone_repo(repo_url, timeout=180)
        print(f"Repository location: {repo_path}")
    except Exception as exc:
        logger.error(f"Failed to clone repository: {exc}")
        return 1
    
    # Verify repository has basic files
    has_basic_files = any(
        (repo_path / name).exists() 
        for name in ("README.md", "setup.py", "pyproject.toml", "package.json")
    )
    
    if has_basic_files:
        print("✓ Basic repository files detected")
    else:
        print("⚠ Warning: Common repository files not found")
    
    # Step 2: Index repository files
    try:
        indexed = get_index(str(repo_path))
        print(f"✓ Indexed {len(indexed)} files")
        
        if not indexed:
            print("⚠ No files to process")
            return 1
            
        # Show sample of indexed files
        sample_size = min(10, len(indexed))
        print(f"\nSample files ({sample_size}/{len(indexed)}):")
        for entry in indexed[:sample_size]:
            print(f"  • {entry.get('file_path', 'unknown')}")
            
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1
    
    # Step 3: Generate code skeleton
    try:
        skeleton_text = code_skeleton(indexed, root_dir=str(repo_path))
        
        # Show preview of skeleton
        lines = skeleton_text.splitlines()
        preview_lines = min(40, len(lines))
        print(f"\n{'='*60}")
        print(f"Skeleton Preview (first {preview_lines} lines)")
        print('='*60)
        print("\n".join(lines[:preview_lines]))
        
        if len(lines) > preview_lines:
            print(f"\n... ({len(lines) - preview_lines} more lines)")
        
        # Save skeleton to file
        skeleton_file = repo_path / "SKELETON.txt"
        skeleton_file.write_text(skeleton_text, encoding="utf-8")
        print(f"\n✓ Skeleton saved to: {skeleton_file}")
        print(f"  Total size: {len(skeleton_text):,} characters, {len(lines):,} lines")
        
    except Exception as e:
        logger.error(f"Skeleton generation failed: {e}")
        return 1
    
    print(f"\n✓ Successfully generated skeleton for {repo_url}")
    return 0


def main() -> int:
    """Main entry point."""
    # Check required environment variable
    if "DATA_DIR" not in os.environ:
        print("Error: DATA_DIR environment variable must be set", file=sys.stderr)
        print("Example: export DATA_DIR=/path/to/data", file=sys.stderr)
        return 1
    
    # Get repository URL from CLI argument (required)
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Repository URL is required.\n"
            "Usage: python3 scripts/generate_skeleton.py <github_url>"
        )
    
    repo_url = sys.argv[1]
    
    # Validate GitHub URL
    if "github.com" not in repo_url:
        print(f"Error: Only GitHub URLs are supported (got: {repo_url})", file=sys.stderr)
        return 1
    
    return asyncio.run(generate_skeleton(repo_url))


if __name__ == "__main__":
    raise SystemExit(main())
