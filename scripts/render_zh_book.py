#!/usr/bin/env python3
"""Render the translated Chinese Quarto sources.

The upstream book configuration expects chapters under the repository root.
This helper builds a temporary Quarto project, overlays files from zh/, renders
the translated Chinese book or selected chapters, and copies the generated site
to zh/_book.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_FILES = [
    "_quarto.yml",
    "_quarto-deploy.yml",
    "references.bib",
    "index.qmd",
    "license.qmd",
]
ROOT_DIRS = ["src", "resources", "_extensions", "animations"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render translated zh/*.qmd sources with Quarto.")
    parser.add_argument(
        "chapters",
        nargs="*",
        help="Chapter basenames or paths to render, for example chap4 or src/chap4.qmd. Omit to render the full book.",
    )
    parser.add_argument("--to", default="html", help="Quarto output format, usually html.")
    parser.add_argument("--out-dir", default="zh/_book", help="Directory to receive rendered output.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary Quarto project for debugging.")
    parser.add_argument("--quarto", default=None, help="Path to quarto executable.")
    parser.add_argument(
        "--python",
        default=os.environ.get("QUARTO_PYTHON"),
        help="Python executable for Quarto kernels. Defaults to QUARTO_PYTHON if set.",
    )
    return parser.parse_args()


def find_quarto(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("quarto")
    if found:
        return found
    windows_default = Path(r"C:\Program Files\Quarto\bin\quarto.exe")
    if windows_default.exists():
        return str(windows_default)
    raise RuntimeError("Could not find quarto. Install Quarto or pass --quarto.")


def normalize_chapters(chapters: list[str]) -> list[str]:
    if not chapters:
        return []
    result: list[str] = []
    for chapter in chapters:
        path = Path(chapter)
        if path.suffix != ".qmd":
            path = Path("src") / f"{chapter}.qmd"
        result.append(path.as_posix())
    return result


def copy_project(repo: Path, temp: Path) -> None:
    for name in ROOT_FILES:
        src = repo / name
        if src.exists():
            shutil.copy2(src, temp / name)
    for name in ROOT_DIRS:
        src = repo / name
        if src.exists():
            shutil.copytree(src, temp / name)


def overlay_zh(repo: Path, temp: Path) -> None:
    zh = repo / "zh"
    for name in ["index.qmd", "license.qmd"]:
        src = zh / name
        if src.exists():
            shutil.copy2(src, temp / name)
    zh_src = zh / "src"
    if zh_src.exists():
        for src in zh_src.glob("*.qmd"):
            shutil.copy2(src, temp / "src" / src.name)


def copy_book(temp: Path, out_dir: Path) -> None:
    book = temp / "_book"
    if not book.exists():
        raise RuntimeError(f"Quarto did not create {book}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in book.iterdir():
        target = out_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    quarto = find_quarto(args.quarto)
    chapters = normalize_chapters(args.chapters)
    temp_path = Path(tempfile.mkdtemp(prefix="aims-zh-render-"))

    try:
        copy_project(repo, temp_path)
        overlay_zh(repo, temp_path)

        env = os.environ.copy()
        if args.python:
            env["QUARTO_PYTHON"] = args.python

        if chapters:
            for chapter in chapters:
                print(f"Rendering {chapter}...")
                command = [quarto, "render", chapter, "--to", args.to]
                subprocess.run(command, cwd=temp_path, env=env, check=True)
        else:
            print("Rendering translated book...")
            command = [quarto, "render", "--to", args.to]
            subprocess.run(command, cwd=temp_path, env=env, check=True)

        copy_book(temp_path, repo / args.out_dir)
        print(f"Rendered output copied to {args.out_dir}")
        if args.keep_temp:
            print(f"Temporary project kept at {temp_path}")
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
