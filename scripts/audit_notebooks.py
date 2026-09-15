"""Audit curated notebooks for structure, portability, and code-cell syntax.

This audit is intentionally conservative: notebook magics and shell commands are
reported separately, while ordinary Python cells are parsed with ``ast``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", re.MULTILINE)
PATH_RE = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/][^\"'\n]+|/content/drive/[^\"'\n]+|/home/[^\"'\n]+)",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[=:]",
    re.IGNORECASE,
)


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def audit(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    issues: list[str] = []
    imports: Counter[str] = Counter()

    for index, cell in enumerate(code):
        text = source_text(cell)
        imports.update(match.split(".")[0] for match in IMPORT_RE.findall(text))
        if cell.get("outputs"):
            issues.append(f"cell {index}: saved output")
        if cell.get("execution_count") is not None:
            issues.append(f"cell {index}: execution count")
        if PATH_RE.search(text):
            issues.append(f"cell {index}: machine-specific path")
        if SECRET_RE.search(text):
            issues.append(f"cell {index}: secret-like assignment")

        stripped = text.lstrip()
        if not stripped or stripped.startswith(("%", "!", "?")):
            continue
        filtered = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        if not filtered.strip():
            continue
        try:
            ast.parse(filtered)
        except SyntaxError as exc:
            issues.append(f"cell {index}: Python syntax error at line {exc.lineno}: {exc.msg}")

    return {
        "cells": len(cells),
        "code": len(code),
        "markdown": len(markdown),
        "imports": sorted(imports),
        "issues": issues,
    }


def main() -> int:
    paths = [Path(value).resolve() for value in sys.argv[1:]]
    if not paths:
        paths = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    if not paths:
        print("No notebooks found.")
        return 1

    issue_count = 0
    for path in paths:
        result = audit(path)
        issue_count += len(result["issues"])
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = path
        print(
            f"{relative}: {result['cells']} cells "
            f"({result['code']} code, {result['markdown']} markdown)"
        )
        print(f"  imports: {', '.join(result['imports']) or 'none'}")
        for issue in result["issues"]:
            print(f"  - {issue}")

    print(f"Audited {len(paths)} notebooks; found {issue_count} issue(s).")
    return 1 if issue_count else 0


if __name__ == "__main__":
    sys.exit(main())
