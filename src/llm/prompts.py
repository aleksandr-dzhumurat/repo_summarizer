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

Below is a code skeleton extracted from the repository. The skeleton starts with a PACKAGES header listing all external dependencies, followed by documentation files with docstrings, and then Python source files with their docstrings.

Format:
- PACKAGES: comma-separated list of external package names (extracted from import statements)
- Documentation: [filename] with extracted docstring content
- File: [filepath] with extracted docstring content (if present)

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
- "summary": Must be 2-3 sentences describing the project's purpose. Example: "**TimesFM** is a pretrained time-series foundation model developed by Google Research for time-series forecasting. It provides JAX and PyTorch implementations with support for fine-tuning via LoRA and DoRA adapters."
- "technologies": Array of strings listing main technologies, languages, frameworks, and libraries used. Extract from PACKAGES header and codebase patterns. Include major frameworks (e.g., "JAX", "PyTorch", "Flax", "TensorFlow").
- "structure": String describing directory layout and code organization. Example: "The project is organized with source code in `src/timesfm/` (core models), `v1/` (legacy implementations), experiments in `v1/experiments/`, and fine-tuning utilities in `v1/peft/` and `src/finetuning/`."

Return ONLY valid JSON, no additional text or markdown formatting.
"""


__all__ = [
    "repo_summarizer_prompt",
]
