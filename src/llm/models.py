"""Pydantic models for structured LLM outputs."""

from typing import Optional

from pydantic import BaseModel, Field


class ComponentInfo(BaseModel):
    """Information about a code component."""
    name: str = Field(..., description="Component name")
    role: str = Field(..., description="Component role in the system")


class RepositorySummary(BaseModel):
    """Structured summary of a GitHub repository."""
    summary: str = Field(
        ...,
        description="Concise overview of the repository's purpose and key features"
    )
    technologies: list[str] = Field(
        ...,
        description="List of primary technologies and frameworks used"
    )
    structure: str = Field(
        ...,
        description="Description of the project structure and architecture"
    )
    key_components: Optional[list[ComponentInfo]] = Field(
        default=None,
        description="Main components or modules and their roles"
    )
    design_patterns: Optional[list[str]] = Field(
        default=None,
        description="Design patterns identified in the codebase"
    )


def pydantic_to_json_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic model to JSON Schema for API requests."""
    schema = model.model_json_schema()
    # Ensure schema conforms to OpenAI format
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


__all__ = [
    "ComponentInfo",
    "RepositorySummary",
    "pydantic_to_json_schema",
]
