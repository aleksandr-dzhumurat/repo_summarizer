"""Generate a naive code skeleton from an index of files.

Function `code_skeleton(index: list)` reads .py files from the provided
index (list of dicts with key `file_path`) and returns a string that
contains the imports found in each file (excluding standard library)
and a list of functions defined there (with parameter lists and docstrings).
Class methods are reported as `ClassName.method`.

Files or functions whose names start with `_` are skipped.
"""

import ast
from pathlib import Path
from typing import List

# Common stdlib modules (Python 3.8+)
STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit", "audioop",
    "base64", "bdb", "binascii", "binhex", "bisect", "builtins", "bz2",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs",
    "codeop", "collections", "colorsys", "compileall", "concurrent", "configparser",
    "contextlib", "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
    "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
    "dis", "distutils", "doctest", "dummy_threading", "email", "encodings",
    "ensurepip", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "msilib", "msvcrt", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "parser", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics",
    "string", "stringprep", "struct", "subprocess", "sunau", "symbol", "symtable",
    "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
    "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
    "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
}


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


def _collect_imports(tree: ast.AST) -> List[str]:
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if _is_stdlib_module(n.name):
                    continue
                if n.asname:
                    imports.append(f"import {n.name} as {n.asname}")
                else:
                    imports.append(f"import {n.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_stdlib_module(module):
                continue
            names = ", ".join([f"{n.name}" + (f" as {n.asname}" if n.asname else "") for n in node.names])
            level = "." * node.level if getattr(node, "level", 0) else ""
            imports.append(f"from {level}{module} import {names}")
    return imports


def code_skeleton(index: List[dict], root_dir: str | None = None) -> str:
    """Generate skeleton string from index list.

    `index` is a list of dicts like {"file_path": "relative/path.py"}.
    Files that do not end with `.py` are ignored.
    """
    out_lines: List[str] = []

    for entry in index:
        fp = entry.get("file_path")
        if not fp or not fp.endswith(".py"):
            continue

        p = Path(fp)
        # Resolve against provided root_dir if file not absolute or doesn't exist
        if not p.is_absolute() and root_dir:
            p = Path(root_dir) / fp

        if not p.exists():
            # Fallback to cwd relative
            p = Path.cwd() / fp
            if not p.exists():
                out_lines.append(f"# Skipped (missing): {fp}")
                continue

        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception as e:
            out_lines.append(f"# Failed to parse {fp}: {e}")
            continue

        out_lines.append(f"File: {fp}")

        imports = _collect_imports(tree)
        if imports:
            out_lines.append("Imports:")
            for im in sorted(set(imports)):
                out_lines.append(f"  - {im}")
        else:
            out_lines.append("Imports: (none)")

        funcs: List[str] = []

        # Top-level functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_"):
                    continue
                sig = _format_arguments(node.args)
                doc = _short_doc(node)
                funcs.append(f"{name}({sig}) -> {doc}")

            # Classes and their methods
            if isinstance(node, ast.ClassDef):
                cls_name = node.name
                # skip private classes? not requested
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fname = item.name
                        if fname.startswith("_"):
                            continue
                        sig = _format_arguments(item.args)
                        doc = _short_doc(item)
                        funcs.append(f"{cls_name}.{fname}({sig}) -> {doc}")

        if funcs:
            out_lines.append("Functions:")
            for f in funcs:
                out_lines.append(f"  - {f}")
        else:
            out_lines.append("Functions: (none)")

        out_lines.append("")

    return "\n".join(out_lines)


__all__ = ["code_skeleton"]
