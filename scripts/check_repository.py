"""Dependency-free quality checks for the thesis repository."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "CITATION.cff",
    "LICENSE.md",
    "environment.yml",
    "requirements.txt",
    "REPRODUCIBILITY.md",
    "data/README.md",
    "notebooks/README.md",
    "notebooks/manifest.csv",
    "papers/publications.bib",
)
WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?:(?<![A-Z])[A-Z]:\\\\(?:Users\\\\[^\\\\]+\\\\)?|(?<![A-Z])[A-Z]:/Users/[^/]+/|(?<![A-Z])[A-Z]:/)",
    re.IGNORECASE,
)
COLAB_PATH = re.compile(r"/content/drive(?:/|\\\\)", re.IGNORECASE)
SECRET = re.compile(
    r"(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[=:]",
    re.IGNORECASE,
)
MAX_GITHUB_FILE_SIZE = 100 * 1024 * 1024
EMBEDDED_IMAGE_DATA = re.compile(r"data:image/[a-z0-9.+-]+;base64,", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_notebook(path: Path, errors: list[str]) -> None:
    try:
        raw_text = path.read_text(encoding="utf-8")
        notebook = json.loads(raw_text)
    except Exception as exc:
        errors.append(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
        return

    if notebook.get("nbformat") != 4:
        errors.append(f"{path.relative_to(ROOT)} is not notebook format 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            errors.append(f"{path.relative_to(ROOT)} cell {index} contains output")
        if cell.get("execution_count") is not None:
            errors.append(f"{path.relative_to(ROOT)} cell {index} has an execution count")
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        filtered = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(("%", "!", "?"))
        )
        if filtered.strip():
            try:
                ast.parse(filtered)
            except SyntaxError as exc:
                errors.append(
                    f"{path.relative_to(ROOT)} cell {index} has invalid Python "
                    f"at line {exc.lineno}: {exc.msg}"
                )

    if WINDOWS_ABSOLUTE_PATH.search(raw_text) or COLAB_PATH.search(raw_text):
        errors.append(f"{path.relative_to(ROOT)} contains a machine-specific path")
    if SECRET.search(raw_text):
        errors.append(f"{path.relative_to(ROOT)} contains a secret-like assignment")
    if EMBEDDED_IMAGE_DATA.search(raw_text):
        errors.append(f"{path.relative_to(ROOT)} contains embedded image data in cell source")


def check_markdown_links(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for value in MARKDOWN_LINK.findall(text):
        destination = value.strip().split("#", 1)[0]
        if not destination or destination.startswith(("http://", "https://", "mailto:")):
            continue
        target = (path.parent / unquote(destination)).resolve()
        if not target.exists():
            errors.append(f"{path.relative_to(ROOT)} has a broken local link: {value}")


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 6:
        errors.append(f"expected 6 maintained notebooks, found {len(notebooks)}")
    for notebook in notebooks:
        check_notebook(notebook, errors)

    manifest_path = ROOT / "notebooks" / "manifest.csv"
    if manifest_path.is_file():
        seen: set[str] = set()
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
            if len(rows) != len(notebooks):
                errors.append(f"manifest contains {len(rows)} rows for {len(notebooks)} notebooks")
            for row in rows:
                if row["notebook"] in seen:
                    errors.append(f"duplicate manifest target: {row['notebook']}")
                seen.add(row["notebook"])
                curated = ROOT / "notebooks" / row["notebook"]
                if not curated.is_file():
                    errors.append(f"manifest target missing: {curated.name}")
                elif digest(curated) != row["curated_sha256"]:
                    errors.append(f"manifest checksum mismatch: {curated.name}")

    excluded = {
        ".git",
        ".venv",
        "venv",
        "env",
        "tmp",
        "temp",
        "outputs",
        "__pycache__",
        "build",
        "dist",
    }
    for folder, directories, files in os.walk(ROOT):
        directories[:] = [
            name for name in directories if name not in excluded and not name.endswith(".egg-info")
        ]
        for filename in files:
            path = Path(folder) / filename
            if path.stat().st_size >= MAX_GITHUB_FILE_SIZE:
                errors.append(f"file reaches GitHub's 100 MB limit: {path.relative_to(ROOT)}")
            if path.suffix == ".md":
                check_markdown_links(path, errors)
    if any((ROOT / "thesis").rglob("*")):
        errors.append("thesis documents must not be included in the repository")
    extra_notebooks = set((ROOT / "notebooks").rglob("*.ipynb")) - set(notebooks)
    if extra_notebooks:
        errors.append("older/additional notebooks must not be included in the repository")

    if errors:
        print("Repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Repository checks passed: {len(notebooks)} maintained notebooks; "
        "repository files and links are valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
