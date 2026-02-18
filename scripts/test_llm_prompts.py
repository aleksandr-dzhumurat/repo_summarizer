"""Test script to verify LLM prompts return correct schema."""

from src.llm.prompts import repo_summarizer_prompt


def test_prompt_format():
    """Verify the repo_summarizer_prompt generates valid instructions."""
    skeleton_sample = """File: src/requests/__init__.py
Imports:
  - from .api import delete, get, head, options, patch, post, put, request
  - from .models import PreparedRequest, Request, Response
  - from .sessions import Session, session
Functions:
  - check_compatibility(urllib3_version, chardet_version) -> Check library compatibility
  
File: src/requests/api.py
Imports:
  - from . import sessions
Functions:
  - request(method, url, **kwargs) -> Construct and send a request
  - get(url, params, **kwargs) -> Send a GET request
"""
    
    prompt = repo_summarizer_prompt(skeleton_sample)
    
    # Verify prompt contains the three required fields
    assert '"summary"' in prompt, "Prompt must reference 'summary' field"
    assert '"technologies"' in prompt, "Prompt must reference 'technologies' field"
    assert '"structure"' in prompt, "Prompt must reference 'structure' field"
    
    # Verify prompt mentions the example response schema
    assert "Requests" in prompt, "Prompt should contain example"
    assert "urllib3" in prompt or "HTTP" in prompt, "Prompt should hint at technologies"
    
    print("✓ Prompt format test passed")
    print("\n--- Sample Prompt Output ---")
    print(prompt[:500] + "...")


if __name__ == "__main__":
    test_prompt_format()
