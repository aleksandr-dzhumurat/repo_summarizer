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


@dataclass
class Part:
    type: str  # 'import_relative', 'import_absolute', 'docstring', 'function', or 'class'
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


def process_code(fp: str, root_dir: str | None = None) -> List[Row]:
    """Process a single Python file and extract its structure.
    
    Args:
        fp: File path (relative or absolute)
        root_dir: Root directory to resolve relative paths
        
    Returns:
        List of output lines for this file
    """
    rows: List[Row] = []
    p = Path(fp)
    skipped_dirs = get_skipped_dirs()
    if any(any(skip_dir in part.lower() for skip_dir in skipped_dirs) for part in p.parts):
        return rows
    
    # Resolve against provided root_dir if file not absolute or doesn't exist
    if not p.is_absolute() and root_dir:
        p = Path(root_dir) / fp

    if not p.exists():
        p = Path.cwd() / fp
        if not p.exists():
            rows.append(Row(source=fp, file_type="code", content=[Part(type="docstring", text=f"# Skipped (missing): {fp}")]))
            return rows
    
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception as e:
        rows.append(Row(source=fp, file_type="code", content=[Part(type="docstring", text=f"# Failed to parse {fp}: {e}")]))
        return rows
    content: List[Part] = []

    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        first_line = module_docstring.strip().splitlines()[0]
        if len(first_line) > 200:
            content.append(Part(type="docstring", text=f"{first_line[:197]}..."))
        else:
            content.append(Part(type="docstring", text=first_line))

    imports = _collect_imports(tree)
    rels = sorted(set(imports["relative"]))
    abss = sorted(set(imports["absolute"]))
    for im in rels:
        content.append(Part(type="import_relative", text=im))
    for im in abss:
        content.append(Part(type="import_absolute", text=im))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_"):
                continue
            sig = _format_arguments(node.args)
            doc = _short_doc(node)
            content.append(Part(type="function", text=f"{name}({sig}) -> {doc}"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls_name = node.name
            if cls_name.startswith("_"):
                continue
            cls_doc = _short_doc(node)
            if cls_doc:
                content.append(Part(type="class", text=f"{cls_name}: {cls_doc}"))
            else:
                content.append(Part(type="class", text=f"{cls_name}"))

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fname = item.name
                    if fname.startswith("_"):
                        continue
                    sig = _format_arguments(item.args)
                    doc = _short_doc(item)
                    content.append(Part(type="function", text=f"{cls_name}.{fname}({sig}) -> {doc}"))

    rows.append(Row(source=fp, file_type="code", content=content))
    return rows


def process_doc(fp: str, root_dir: str | None = None) -> List[Row]:
    """Process a single documentation file (.md, .rst, .txt).
    
    Args:
        fp: File path (relative or absolute)
        root_dir: Root directory to resolve relative paths
        
    Returns:
        List of output lines for this file
    """
    rows: List[Row] = []
    p = Path(fp)
    
    # Skip LICENSE, SKELETON, CODEOWNERS, SECURITY, and REQUIREMENTS files
    if p.name.upper().startswith(('LICENSE', 'SKELETON', 'CODEOWNERS', 'SECURITY', 'REQUIREMENTS')):
        return rows
    
    skipped_dirs = get_skipped_dirs()
    if any(any(skip_dir in part.lower() for skip_dir in skipped_dirs) for part in p.parts):
        return rows
    
    # Resolve against provided root_dir if file not absolute or doesn't exist
    if not p.is_absolute() and root_dir:
        p = Path(root_dir) / fp

    if not p.exists():
        p = Path.cwd() / fp
        if not p.exists():
            rows.append(Row(source=fp, file_type="doc", content=[Part(type="docstring", text=f"# Skipped (missing): {fp}")]))
            return rows
    
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        # Clean markdown content (remove links, HTML tags, code blocks)
        cleaned = clean_markdown_text(content)
        
        # Add cleaned content (truncated to first 500 chars for brevity)
        if cleaned:
            preview = cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
            rows.append(Row(source=fp, file_type="doc", content=[Part(type="docstring", text=preview)]))
        else:
            rows.append(Row(source=fp, file_type="doc", content=[]))
    except Exception as e:
        rows.append(Row(source=fp, file_type="doc", content=[Part(type="docstring", text=f"# Failed to read {fp}: {e}")]))
    
    return rows


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


def filter_rows(rows: List[Row]) -> List[Row]:
    include_functions = get_functions_flag()
    include_classes = get_class_definitions_flag()
    include_methods = get_class_methods_flag()
    include_rel = get_relative_imports_flag()
    include_abs = get_absolute_imports_flag()
    filtered: List[Row] = []
    for row in rows:
        if row.file_type != "code":
            filtered.append(row)
            continue

        new_parts: List[Part] = []
        for part in row.content:
            if part.type == "import_relative" and not include_rel:
                continue
            if part.type == "import_absolute" and not include_abs:
                continue
            if part.type == "function":
                is_method = "." in part.text.split("(")[0]
                if is_method and not include_methods:
                    continue
                if not is_method and not include_functions:
                    continue
            elif part.type == "class" and not include_classes:
                continue
            new_parts.append(part)

        filtered.append(Row(source=row.source, file_type=row.file_type, content=new_parts))
    return filtered


def _render_rows(rows: List[Row], header: str = "") -> str:
    lines: List[str] = []
    if header:
        lines.append(header)
        lines.append("")
    for row in rows:
        if row.content and ': (none)' in row.content[0].text:
            continue
            
        if row.file_type == "code":
            lines.append(f"File: {row.source}")
        else:
            lines.append(f"Documentation: {row.source}")

        docstrings = [p.text for p in row.content if p.type == "docstring"]
        imports = [p.text for p in row.content if p.type in ("import_relative", "import_absolute")]
        functions = [p.text for p in row.content if p.type == "function"]
        classes = [p.text for p in row.content if p.type == "class"]

        if docstrings:
            lines.append("Docstring:")
            for text in docstrings:
                lines.append(f"  {text}")

        if imports:
            lines.append("Imports:")
            for im in imports:
                lines.append(f"  - {im}")

        if functions:
            lines.append("Functions:")
            for f in functions:
                lines.append(f"  - {f}")

        if classes:
            lines.append("Classes:")
            for c in classes:
                lines.append(f"  - {c}")

        lines.append("")
    return "\n".join(lines)


def code_skeleton(index: List[dict], root_dir: str | None = None) -> str:
    """Generate skeleton string from index list.

    `index` is a list of dicts like {"file_path": "relative/path.py"}.
    Python files (.py) are processed with process_code().
    Documentation files (.md, .rst, .txt) are processed with process_doc().
    """
    rows: List[Row] = []
    py_entries = 0
    code_rows = 0

    for entry in index:
        fp = entry.get("file_path")
        if not fp:
            continue
        
        if fp.endswith(".py"):
            py_entries += 1
            code_result = process_code(fp, root_dir)
            code_rows += len(code_result)
            rows.extend(code_result)
        elif fp.endswith((".md", ".rst", ".txt")):
            rows.extend(process_doc(fp, root_dir))
        # Other file types are silently skipped

    header_parts = [p for row in rows for p in row.content if p.type == "import_absolute"]
    header = import_summarize(header_parts)
    rows = filter_rows(rows)
    return _render_rows(rows, header=header)


__all__ = ["code_skeleton", "process_code", "process_doc", "Row", "Part"]
