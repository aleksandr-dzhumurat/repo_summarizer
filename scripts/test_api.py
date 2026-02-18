#!/usr/bin/env python3
"""Lightweight API smoke test for the repository summarizer.

This script checks `/health` and submits a `/summarize` request.
It does NOT poll for results (the service no longer exposes `/info`).
"""

import asyncio
import json
import os

import httpx

BASE_URL = os.getenv("SUMMARIZER_BASE", "http://localhost:8000")


async def test_api():
    async with httpx.AsyncClient(timeout=30) as client:
        # Health
        print("1. Testing /health")
        r = await client.get(f"{BASE_URL}/health")
        print(r.status_code, r.json())

        # Summarize (do not poll for completion)
        print("2. Submitting /summarize request (no polling)")
        payload = {"github_url": "https://github.com/psf/requests"}
        r = await client.post(f"{BASE_URL}/summarize", json=payload)
        print(r.status_code)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        print(json.dumps(body, indent=2))


if __name__ == "__main__":
    asyncio.run(test_api())
