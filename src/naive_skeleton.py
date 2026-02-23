"""Generate a naive code skeleton from an index of files.

Function `code_skeleton(index: list)` reads .py files from the provided
index (list of dicts with key `file_path`) and returns a string that
contains the imports found in each file (excluding standard library)
and a list of functions defined there (with parameter lists and docstrings).
Class methods are reported as `ClassName.method`.

Files or functions whose names start with `_` are skipped.
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from .utils import (
    clean_markdown_text,
    get_absolute_imports_flag,
    get_class_definitions_flag,
    get_class_methods_flag,
    get_functions_flag,
    get_relative_imports_flag,
    get_skipped_dirs,
)

# Cache stdlib modules once at module load time (Python 3.10+)
STDLIB_MODULES: Set[str] = sys.stdlib_module_names

# Part type constants - Simple is better than complex
PART_IMPORT_REL = "import_relative"
PART_IMPORT_ABS = "import_absolute"
PART_DOCSTRING = "docstring"
PART_FUNCTION = "function"
PART_CLASS = "class"


@dataclass
class Part:
    type: str  # PART_* constant
    text: str


@dataclass
class Row:
    source: str
    file_type: str  # 'doc' or 'code'
    content: List[Part]


def _format_arguments(args: ast.arguments) -> str:
    parts: List[str] = []

    def _arg_name(a: ast.arg) -> str:
        return a.arg

    # positional-only (Python 3.8+)
    posonly = getattr(args, "posonlyargs", [])
    for a in posonly:
        parts.append(_arg_name(a))
    if posonly:
        parts.append("/")

    # regular args
    for a in args.args:
        parts.append(_arg_name(a))

    # vararg
    if args.vararg:
        parts.append("*" + _arg_name(args.vararg))

    # keyword-only args
    for a in args.kwonlyargs:
        parts.append(_arg_name(a))

    # kwarg
    if args.kwarg:
        parts.append("**" + _arg_name(args.kwarg))

    # Note: defaults and annotations are omitted for brevity; names are primary
    return ", ".join(parts)


def _short_doc(node: ast.AST, max_len: int = 120) -> str:
    doc = ast.get_docstring(node) or ""
    if not doc:
        return ""
    one = doc.strip().splitlines()[0]
    return (one[: max_len - 3] + "...") if len(one) > max_len else one


def _is_stdlib_module(module_name: str) -> bool:
    """Check if a module name is from the Python standard library."""
    top_level = module_name.split(".")[0]
    return top_level in STDLIB_MODULES


def _resolve_file_path(fp: str, root_dir: str | None) -> Path | None:
    """Resolve file path, trying root_dir and cwd. Returns None if not found."""
    p = Path(fp)
    if p.exists():
        return p
    
    if not p.is_absolute() and root_dir:
        p = Path(root_dir) / fp
        if p.exists():
            return p
    
    p = Path.cwd() / fp
    return p if p.exists() else None


def _should_skip_by_dir(path: Path) -> bool:
    """Check if path contains any skipped directory."""
    skipped_dirs = get_skipped_dirs()
    return any(any(skip_dir in part.lower() for skip_dir in skipped_dirs) for part in path.parts)


def _collect_imports(tree: ast.AST) -> dict:
    """Return dict with 'relative' and 'absolute' import lists."""
    rel_imports: List[str] = []
    abs_imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if _is_stdlib_module(n.name):
                    continue
                if n.asname:
                    abs_imports.append(f"import {n.name} as {n.asname}")
                else:
                    abs_imports.append(f"import {n.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_stdlib_module(module):
                continue
            names = ", ".join([f"{n.name}" + (f" as {n.asname}" if n.asname else "") for n in node.names])
            level = getattr(node, "level", 0)
            if level > 0:
                rel_imports.append(f"from {'.'*level}{module} import {names}")
            else:
                abs_imports.append(f"from {module} import {names}")
    return {"relative": rel_imports, "absolute": abs_imports}


def _extract_module_docstring(tree: ast.AST) -> Part | None:
    """Extract module-level docstring as a Part."""
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    first_line = doc.strip().splitlines()[0]
    text = f"{first_line[:197]}..." if len(first_line) > 200 else first_line
    return Part(type=PART_DOCSTRING, text=text)


def _extract_imports(tree: ast.AST) -> List[Part]:
    """Extract import statements as Parts."""
    imports = _collect_imports(tree)
    parts: List[Part] = []
    for im in sorted(set(imports["relative"])):
        parts.append(Part(type=PART_IMPORT_REL, text=im))
    for im in sorted(set(imports["absolute"])):
        parts.append(Part(type=PART_IMPORT_ABS, text=im))
    return parts


def _extract_functions(tree: ast.AST) -> List[Part]:
    """Extract top-level functions as Parts."""
    parts: List[Part] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                sig = _format_arguments(node.args)
                doc = _short_doc(node)
                parts.append(Part(type=PART_FUNCTION, text=f"{node.name}({sig}) -> {doc}"))
    return parts


def _extract_classes(tree: ast.AST) -> List[Part]:
    """Extract class definitions and their methods as Parts."""
    parts: List[Part] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            
            cls_doc = _short_doc(node)
            cls_text = f"{node.name}: {cls_doc}" if cls_doc else node.name
            parts.append(Part(type=PART_CLASS, text=cls_text))
            
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        sig = _format_arguments(item.args)
                        doc = _short_doc(item)
                        parts.append(Part(type=PART_FUNCTION, text=f"{node.name}.{item.name}({sig}) -> {doc}"))
    return parts


def process_code(fp: str, root_dir: str | None = None) -> List[Row]:
    """Process a single Python file and extract its structure.
    
    Args:
        fp: File path (relative or absolute)
        root_dir: Root directory to resolve relative paths
        
    Returns:
        List of output lines for this file
    """
    p = Path(fp)
    if _should_skip_by_dir(p):
        return []
    
    resolved_path = _resolve_file_path(fp, root_dir)
    if not resolved_path:
        return [Row(source=fp, file_type="code", content=[Part(type=PART_DOCSTRING, text=f"# Skipped (missing): {fp}")])]
    
    try:
        src = resolved_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception as e:
        return [Row(source=fp, file_type="code", content=[Part(type=PART_DOCSTRING, text=f"# Failed to parse {fp}: {e}")])]
    
    content: List[Part] = []

    # Flat is better than nested - collect all parts
    if doc_part := _extract_module_docstring(tree):
        content.append(doc_part)
    
    content.extend(_extract_imports(tree))
    content.extend(_extract_functions(tree))
    content.extend(_extract_classes(tree))

    return [Row(source=fp, file_type="code", content=content)]


def process_doc(fp: str, root_dir: str | None = None) -> List[Row]:
    """Process a single documentation file (.md, .rst, .txt).
    
    Args:
        fp: File path (relative or absolute)
        root_dir: Root directory to resolve relative paths
        
    Returns:
        List of output lines for this file
    """
    p = Path(fp)
    
    # Skip system/config files
    if p.name.upper().startswith(('LICENSE', 'SKELETON', 'CODEOWNERS', 'SECURITY', 'REQUIREMENTS')):
        return []
    
    if _should_skip_by_dir(p):
        return []
    
    resolved_path = _resolve_file_path(fp, root_dir)
    if not resolved_path:
        return [Row(source=fp, file_type="doc", content=[Part(type=PART_DOCSTRING, text=f"# Skipped (missing): {fp}")])]
    
    try:
        content = resolved_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_markdown_text(content)
        
        if not cleaned:
            return [Row(source=fp, file_type="doc", content=[])]
        
        preview = cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
        return [Row(source=fp, file_type="doc", content=[Part(type=PART_DOCSTRING, text=preview)])]
    except Exception as e:
        return [Row(source=fp, file_type="doc", content=[Part(type=PART_DOCSTRING, text=f"# Failed to read {fp}: {e}")])]


def import_summarize(import_parts: List[Part]) -> str:
    packages: set[str] = set()
    for part in import_parts:
        text = part.text.strip()
        if text.startswith("import "):
            name = text[len("import "):].split(" as ")[0].strip()
        elif text.startswith("from "):
            name = text[len("from "):].split(" import ")[0].strip()
        else:
            name = text
        if name:
            packages.add(name.split(".")[0])
    summary = ", ".join(sorted(packages))
    return f"PACKAGES: {summary}" if summary else ""


def _should_include_part(part: Part, flags: dict) -> bool:
    """Check if a part should be included based on config flags."""
    if part.type == PART_IMPORT_REL:
        return flags["include_rel"]
    if part.type == PART_IMPORT_ABS:
        return flags["include_abs"]
    if part.type == PART_FUNCTION:
        is_method = "." in part.text.split("(")[0]
        return flags["include_methods"] if is_method else flags["include_functions"]
    if part.type == PART_CLASS:
        return flags["include_classes"]
    return True  # docstrings always included


def filter_rows(rows: List[Row]) -> List[Row]:
    """Filter row content based on configuration flags."""
    flags = {
        "include_functions": get_functions_flag(),
        "include_classes": get_class_definitions_flag(),
        "include_methods": get_class_methods_flag(),
        "include_rel": get_relative_imports_flag(),
        "include_abs": get_absolute_imports_flag(),
    }
    
    filtered: List[Row] = []
    for row in rows:
        if row.file_type != "code":
            filtered.append(row)
            continue
        
        new_parts = [p for p in row.content if _should_include_part(p, flags)]
        filtered.append(Row(source=row.source, file_type=row.file_type, content=new_parts))
    
    return filtered


def _render_rows(rows: List[Row], header: str = "") -> str:
    """Render rows into formatted text output."""
    lines: List[str] = []
    if header:
        lines.extend([header, ""])
    
    for row in rows:
        # Skip rows with placeholder content
        if row.content and ': (none)' in row.content[0].text:
            continue
        
        # Add file header
        prefix = "File" if row.file_type == "code" else "Documentation"
        lines.append(f"{prefix}: {row.source}")
        
        # Group parts by type
        by_type = {
            PART_DOCSTRING: [],
            "imports": [],
            PART_FUNCTION: [],
            PART_CLASS: [],
        }
        
        for p in row.content:
            if p.type == PART_DOCSTRING:
                by_type[PART_DOCSTRING].append(p.text)
            elif p.type in (PART_IMPORT_REL, PART_IMPORT_ABS):
                by_type["imports"].append(p.text)
            elif p.type == PART_FUNCTION:
                by_type[PART_FUNCTION].append(p.text)
            elif p.type == PART_CLASS:
                by_type[PART_CLASS].append(p.text)
        
        # Render each section
        if by_type[PART_DOCSTRING]:
            lines.append("Docstring:")
            lines.extend(f"  {text}" for text in by_type[PART_DOCSTRING])
        
        if by_type["imports"]:
            lines.append("Imports:")
            lines.extend(f"  - {im}" for im in by_type["imports"])
        
        if by_type[PART_FUNCTION]:
            lines.append("Functions:")
            lines.extend(f"  - {f}" for f in by_type[PART_FUNCTION])
        
        if by_type[PART_CLASS]:
            lines.append("Classes:")
            lines.extend(f"  - {c}" for c in by_type[PART_CLASS])
        
        lines.append("")
    
    return "\n".join(lines)


def code_skeleton(index: List[dict], root_dir: str | None = None) -> str:
    """Generate skeleton string from index list.

    `index` is a list of dicts like {"file_path": "relative/path.py"}.
    Python files (.py) are processed with process_code().
    Documentation files (.md, .rst, .txt) are processed with process_doc().
    """
    rows: List[Row] = []

    for entry in index:
        fp = entry.get("file_path")
        if not fp:
            continue
        
        if fp.endswith(".py"):
            rows.extend(process_code(fp, root_dir))
        elif fp.endswith((".md", ".rst", ".txt")):
            rows.extend(process_doc(fp, root_dir))
        # Other file types are silently skipped

    # Generate header from absolute imports before filtering
    header_parts = [p for row in rows for p in row.content if p.type == PART_IMPORT_ABS]
    header = import_summarize(header_parts)
    
    # Apply filtering and render
    rows = filter_rows(rows)
    return _render_rows(rows, header=header)


__all__ = ["code_skeleton", "process_code", "process_doc", "Row", "Part"]
