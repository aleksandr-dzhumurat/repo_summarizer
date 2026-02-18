"""Repository summarization prompts for DeepSeek LLM."""


def repo_summarizer_prompt(skeleton_text: str) -> str:
    """
    Generate a repository summarization prompt based on code skeleton.
    
    Args:
        skeleton_text: Code skeleton extracted from the repository (imports + functions).
    
    Returns:
        A formatted prompt string for the LLM.
    """
    return f"""You are an expert software engineer tasked with analyzing and summarizing a GitHub Python repository.

Below is a code skeleton extracted from the repository showing:
- File paths and imports (excluding standard library)
- Top-level functions and class methods with signatures and docstrings

Analyze this structure and provide a concise summary.

--- CODE SKELETON ---
{skeleton_text}

--- ANALYSIS TASK ---

Based on the code skeleton above, provide a JSON response with the following structure:
{{
  "summary": "A human-readable description of what the project does (2-3 sentences). Be specific about the project's purpose and main functionality.",
  "technologies": ["Python", "technology1", "technology2", ...],
  "structure": "Brief description of the project structure, including main directories and how code is organized."
}}

Response requirements:
- "summary": Must be 2-3 sentences describing the project's purpose. Example: "**Requests** is a popular Python library for making HTTP requests. It provides a simple, elegant API for handling HTTP operations..."
- "technologies": Array of strings listing main technologies, languages, frameworks, and libraries used. Extract from imports and codebase patterns.
- "structure": String describing directory layout and code organization. Example: "The project follows a standard Python package layout with the main source code in `src/requests/`, tests in `tests/`, and documentation in `docs/`."

Return ONLY valid JSON, no additional text or markdown formatting.
"""


__all__ = [
    "repo_summarizer_prompt",
]
