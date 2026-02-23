#!/usr/bin/env python3
"""Lightweight API smoke test for the repository summarizer.

Tests /health and /summarize endpoints with response time measurement.
"""

import json
import os
import time

import httpx

BASE_URL = os.getenv("SUMMARIZER_BASE", "http://localhost:8000")


def format_time(elapsed_ms: float) -> str:
    """Format elapsed time in milliseconds, converting to seconds if >= 1000."""
    if elapsed_ms >= 1000:
        return f"{elapsed_ms / 1000:.2f} s"
    return f"{elapsed_ms:.2f} ms"


def test_api():
    """Test API endpoints with synchronous requests."""
    with httpx.Client(timeout=300) as client:
        # Health check
        print("1. Testing /health")
        start = time.perf_counter()
        r = client.get(f"{BASE_URL}/health")
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{r.status_code} {r.json()} ({format_time(elapsed_ms)})")

        # Summarize request
        print("\n2. Testing /summarize")
        payload = {"github_url": "https://github.com/psf/requests"}
        start = time.perf_counter()
        r = client.post(f"{BASE_URL}/summarize", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"Status: {r.status_code}")
        print(f"Response time: {format_time(elapsed_ms)}")
        
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        print(json.dumps(body, indent=2))


if __name__ == "__main__":
    test_api()

