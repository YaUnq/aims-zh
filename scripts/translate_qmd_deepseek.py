#!/usr/bin/env python3
"""Translate Quarto QMD sources with the DeepSeek chat API.

The script is deliberately conservative: it skips YAML front matter and fenced
code blocks, translates Markdown/Quarto prose in chunks, and caches successful
chunks so interrupted runs can resume without paying for the same text twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_CACHE = ".translation-cache/deepseek-qmd.json"
DEFAULT_GLOBS = ["index.qmd", "license.qmd", "src/*.qmd"]
DEFAULT_EXCLUDES = ["src/_*.qmd"]
PROMPT_VERSION = "2026-05-25-quarto-structure-v2"

SYSTEM_PROMPT = """You are a structure-preserving translator for Quarto Markdown (.qmd) book source files.

Your highest priority is preserving the original Quarto/Pandoc/Markdown structure exactly.
Translate only human-facing English prose into fluent, publication-quality Simplified Chinese.
Do not repair, reformat, normalize, summarize, reorder, merge, split, add, or remove any document structure.
Copy all syntax tokens exactly, including headings and levels, anchors, attributes in {...}, fenced div markers such as :::, code fences, table pipes and delimiter rows, citations, cross references, math, raw HTML, YAML, URLs, paths, commands, package names, identifiers, labels, and model or benchmark names.
Preserve blank lines, leading/trailing whitespace around the fragment, list markers, numbering, and table row counts.
For technical terms, prefer established Chinese translations and include the English term in parentheses when it helps clarity on first use.
Return only the translated Quarto Markdown fragment, with no commentary and no wrapper."""

USER_TEMPLATE = """Translate the SOURCE block from English to Simplified Chinese.

Hard rules:
- You are translating Quarto Markdown (.qmd), not ordinary prose.
- Output only the translated SOURCE content. Do not output BEGIN_SOURCE or END_SOURCE.
- Preserve the original document structure exactly.
- Do not add, remove, merge, split, reorder, or normalize lines that contain structural syntax.
- Structural syntax includes headings with anchors, fenced div lines beginning with :::, code fences, table delimiter rows, table pipes, citations, cross references, math, attributes in {...}, raw HTML, YAML, URLs, file paths, commands, package names, labels, identifiers, and model or benchmark names.
- Translate prose only. Keep syntax, code, citations, anchors, labels, math, and commands unchanged.
- Preserve leading and trailing blank lines.
- Do not wrap the answer in a Markdown code fence.

BEGIN_SOURCE
{text}
END_SOURCE"""

FENCE_RE = re.compile(r"^\s*(```|~~~)")
FRONT_MATTER_RE = re.compile(r"^\s*---\s*$")
PROSE_HINT_RE = re.compile(r"[A-Za-z][A-Za-z]{2,}")
CHUNK_CAPTION_RE = re.compile(
    r"^(\s*#\|\s*(?:fig-cap|tbl-cap|fig-subcap|caption|code-summary)\s*:\s*)(.*?)(\r?\n?)$",
    re.MULTILINE,
)
FENCED_RESPONSE_RE = re.compile(r"^\s*```(?:qmd|markdown|md)?\s*\r?\n(.*?)\r?\n```\s*$", re.DOTALL)


@dataclass(frozen=True)
class Segment:
    text: str
    translatable: bool


@dataclass(frozen=True)
class TextPart:
    text: str
    translatable: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Quarto .qmd files to Simplified Chinese with DeepSeek."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or glob patterns to translate. Defaults to index.qmd, license.qmd, and src/*.qmd.",
    )
    parser.add_argument("--out-dir", default="zh", help="Output directory for translated files.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files instead of writing to --out-dir.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When used with --in-place, write a .bak copy before overwriting.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--api-base", default=DEFAULT_BASE_URL, help="Chat completions endpoint.")
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable that contains the DeepSeek API key.",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="JSON cache file path.")
    parser.add_argument("--max-chars", type=int, default=3000, help="Maximum characters per API chunk.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--retries", type=int, default=4, help="Retries per failed API call.")
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=2,
        help="Retries when a translation breaks Quarto/Markdown structure.",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between API calls in seconds.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent DeepSeek API calls. Start with 3-5 to avoid rate limits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files and chunks without calling the API.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached translations and call the API again.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[Path]:
    patterns = patterns or DEFAULT_GLOBS
    excludes = {Path(p).as_posix() for pattern in DEFAULT_EXCLUDES for p in Path().glob(pattern)}
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        for path in matches:
            if path.is_file() and path.suffix == ".qmd" and path.as_posix() not in excludes:
                files.append(path)
    return sorted(dict.fromkeys(files))


def split_qmd(text: str) -> list[Segment]:
    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    buf: list[str] = []
    translatable = True
    in_fence = False
    in_front_matter = bool(lines and FRONT_MATTER_RE.match(lines[0]))

    def flush() -> None:
        nonlocal buf
        if buf:
            joined = "".join(buf)
            should_translate = translatable and bool(PROSE_HINT_RE.search(joined))
            segments.append(Segment(joined, should_translate))
            buf = []

    for i, line in enumerate(lines):
        if i == 0 and in_front_matter:
            flush()
            translatable = False
            buf.append(line)
            continue

        if in_front_matter:
            buf.append(line)
            if FRONT_MATTER_RE.match(line):
                in_front_matter = False
                flush()
                translatable = True
            continue

        if FENCE_RE.match(line):
            if not in_fence:
                flush()
                in_fence = True
                translatable = False
                buf.append(line)
            else:
                buf.append(line)
                in_fence = False
                flush()
                translatable = True
            continue

        buf.append(line)

    flush()
    return segments


def chunk_segment(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    blocks = re.split(r"(\n\s*\n)", text)
    for block in blocks:
        if current and current_len + len(block) > max_chars:
            chunks.append("".join(current))
            current = []
            current_len = 0
        if len(block) > max_chars:
            lines = block.splitlines(keepends=True)
            for line in lines:
                if current and current_len + len(line) > max_chars:
                    chunks.append("".join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line)
        else:
            current.append(block)
            current_len += len(block)
    if current:
        chunks.append("".join(current))
    return chunks


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache file is not a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def cache_key(model: str, text: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def call_deepseek(
    *,
    api_base: str,
    api_key: str,
    model: str,
    text: str,
    temperature: float,
    retries: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.replace("{text}", text)},
        ],
        "temperature": temperature,
        "stream": False,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(api_base, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = min(2**attempt, 30)
            print(f"API call failed ({exc}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"DeepSeek API call failed after {retries + 1} attempts: {last_error}")


def clean_model_output(text: str) -> str:
    match = FENCED_RESPONSE_RE.match(text)
    if match:
        text = match.group(1)
    text = re.sub(r"^\s*<qmd_fragment>\s*\r?\n?", "", text)
    text = re.sub(r"\r?\n?\s*</qmd_fragment>\s*$", "", text)
    text = re.sub(r"^\s*BEGIN_SOURCE\s*\r?\n?", "", text)
    text = re.sub(r"\r?\n?\s*END_SOURCE\s*$", "", text)
    return text


def preserve_edge_whitespace(source: str, translated: str) -> str:
    source_leading = re.match(r"^\s*", source).group(0)
    source_trailing = re.search(r"\s*$", source).group(0)
    translated = translated.strip()
    return source_leading + translated + source_trailing


def count_table_rows(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("|"))


def count_fenced_div_markers(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(":::"))


def find_anchors(text: str) -> set[str]:
    return set(re.findall(r"\{#[A-Za-z0-9_-]+[^}]*\}", text))


def validate_translation(source: str, translated: str) -> None:
    bad_tokens = ["```qmd", "```markdown", "```md", "``````"]
    if any(token in translated for token in bad_tokens):
        raise ValueError("translation contains an unexpected Markdown code fence wrapper")
    wrapper_markers = ["<qmd_fragment>", "</qmd_fragment>", "BEGIN_SOURCE", "END_SOURCE"]
    if any(marker in translated for marker in wrapper_markers):
        raise ValueError("translation contains prompt wrapper markers")
    if "```" not in source and "```" in translated:
        raise ValueError("translation introduced a code fence into prose")
    if count_fenced_div_markers(source) != count_fenced_div_markers(translated):
        raise ValueError(
            "translation changed fenced div marker count "
            f"({count_fenced_div_markers(source)} -> {count_fenced_div_markers(translated)})"
        )
    if count_table_rows(source) != count_table_rows(translated):
        raise ValueError(
            "translation changed Markdown table row count "
            f"({count_table_rows(source)} -> {count_table_rows(translated)})"
        )
    missing_anchors = find_anchors(source) - find_anchors(translated)
    if missing_anchors:
        raise ValueError(f"translation dropped anchor(s): {', '.join(sorted(missing_anchors))}")


def validate_final_output(text: str) -> None:
    bad_patterns = [
        r"```(?:qmd|markdown|md)\b",
        r"``````",
        r"^```[*_]",
        r"^#{1,6}\s+.*```",
        r"</?qmd_fragment>",
        r"\b(?:BEGIN_SOURCE|END_SOURCE)\b",
    ]
    for pattern in bad_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            line_no = text[: match.start()].count("\n") + 1
            raise RuntimeError(f"Final output contains suspicious Markdown near line {line_no}: {match.group(0)}")


def validate_final_against_source(source: str, translated: str) -> None:
    source_anchors = find_anchors(source)
    translated_anchors = find_anchors(translated)
    missing = source_anchors - translated_anchors
    extra = translated_anchors - source_anchors
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing anchors: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("extra anchors: " + ", ".join(sorted(extra)))
        raise RuntimeError("Final output failed anchor validation; " + "; ".join(parts))


def translate_chunks(
    chunks: list[str],
    *,
    cache: dict[str, str],
    cache_path: Path,
    args: argparse.Namespace,
    api_key: str | None,
) -> tuple[dict[str, str], int]:
    results: dict[str, str] = {}
    pending: dict[str, str] = {}

    for chunk in chunks:
        key = cache_key(args.model, chunk)
        if not args.force and key in cache:
            cached = clean_model_output(cache[key])
            cached = preserve_edge_whitespace(chunk, cached)
            try:
                validate_translation(chunk, cached)
                results[key] = cached
            except ValueError:
                pending.setdefault(key, chunk)
        else:
            pending.setdefault(key, chunk)

    if not pending:
        return results, 0
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env}.")

    def retry_chunks(text: str) -> list[str]:
        target = max(700, min(args.max_chars // 2, len(text) // 2))
        pieces = chunk_segment(text, target)
        if len(pieces) <= 1:
            return []
        return pieces

    def worker(text: str, depth: int = 0) -> str:
        last_error: Exception | None = None
        for attempt in range(args.validation_retries + 1):
            translated = call_deepseek(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                text=text,
                temperature=args.temperature,
                retries=args.retries,
            )
            translated = clean_model_output(translated)
            translated = preserve_edge_whitespace(text, translated)
            try:
                validate_translation(text, translated)
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return translated
            except ValueError as exc:
                last_error = exc
                if attempt < args.validation_retries:
                    print(f"Validation failed ({exc}); retrying chunk...", file=sys.stderr)
                    time.sleep(min(2**attempt, 10))
        smaller_chunks = retry_chunks(text)
        if smaller_chunks and depth < 4:
            print(
                f"Validation failed ({last_error}); splitting chunk into {len(smaller_chunks)} smaller chunk(s).",
                file=sys.stderr,
            )
            return "".join(worker(piece, depth + 1) for piece in smaller_chunks)
        preview = text[:240].replace("\n", "\\n")
        raise RuntimeError(f"Translation failed structural validation: {last_error}. Chunk starts: {preview}")

    calls = 0
    max_workers = max(1, args.concurrency)
    if max_workers == 1:
        for key, chunk in pending.items():
            translated = worker(chunk)
            cache[key] = translated
            results[key] = translated
            calls += 1
            save_cache(cache_path, cache)
            print(f"Translated {calls}/{len(pending)} uncached chunk(s).")
        return results, calls

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(worker, chunk): key
            for key, chunk in pending.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            translated = future.result()
            cache[key] = translated
            results[key] = translated
            calls += 1
            save_cache(cache_path, cache)
            print(f"Translated {calls}/{len(pending)} uncached chunk(s).")

    return results, calls


def split_quoted_value(value: str) -> tuple[str, str, str]:
    stripped = value.strip()
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    core = stripped
    if len(core) >= 2 and core[0] == core[-1] and core[0] in {"'", '"'}:
        return leading + core[0], core[1:-1], core[-1] + trailing
    return leading, core, trailing


def count_translatable_chunk_options(text: str) -> int:
    count = 0
    for match in CHUNK_CAPTION_RE.finditer(text):
        _, value, _ = match.groups()
        _, core, _ = split_quoted_value(value)
        if PROSE_HINT_RE.search(core):
            count += 1
    return count


def split_chunk_options(text: str) -> list[TextPart]:
    parts: list[TextPart] = []
    cursor = 0
    for match in CHUNK_CAPTION_RE.finditer(text):
        if match.start() > cursor:
            parts.append(TextPart(text[cursor:match.start()], False))
        prefix, value, newline = match.groups()
        before, core, after = split_quoted_value(value)
        if not PROSE_HINT_RE.search(core):
            parts.append(TextPart(match.group(0), False))
        else:
            parts.append(TextPart(prefix + before, False))
            parts.append(TextPart(core, True))
            parts.append(TextPart(after + newline, False))
        cursor = match.end()
    if cursor < len(text):
        parts.append(TextPart(text[cursor:], False))
    return parts


def output_path_for(source: Path, out_dir: Path, in_place: bool) -> Path:
    if in_place:
        return source
    return out_dir / source


def translate_file(
    source: Path,
    *,
    out_dir: Path,
    in_place: bool,
    backup: bool,
    cache: dict[str, str],
    cache_path: Path,
    args: argparse.Namespace,
    api_key: str | None,
) -> tuple[int, int]:
    original = source.read_text(encoding="utf-8")
    segments = split_qmd(original)
    chunks_total = sum(
        len(chunk_segment(segment.text, args.max_chars)) for segment in segments if segment.translatable
    ) + sum(count_translatable_chunk_options(segment.text) for segment in segments if not segment.translatable)
    if args.dry_run:
        print(f"{source}: {chunks_total} translatable chunk(s)")
        return chunks_total, 0

    parts: list[TextPart] = []
    for segment in segments:
        if not segment.translatable:
            parts.extend(split_chunk_options(segment.text))
        else:
            parts.extend(TextPart(chunk, True) for chunk in chunk_segment(segment.text, args.max_chars))

    translations, calls = translate_chunks(
        [part.text for part in parts if part.translatable],
        cache=cache,
        cache_path=cache_path,
        args=args,
        api_key=api_key,
    )

    translated_text = "".join(
        translations[cache_key(args.model, part.text)] if part.translatable else part.text
        for part in parts
    )
    validate_final_output(translated_text)
    validate_final_against_source(original, translated_text)
    destination = output_path_for(source, out_dir, in_place)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if in_place and backup:
        backup_path = source.with_suffix(source.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(original, encoding="utf-8")
    destination.write_text(translated_text, encoding="utf-8", newline="")
    print(f"{source} -> {destination} ({chunks_total} chunk(s), {calls} API call(s))")
    return chunks_total, calls


def main() -> int:
    args = parse_args()
    files = expand_paths(args.paths)
    if not files:
        print("No .qmd files matched.", file=sys.stderr)
        return 2

    api_key = None if args.dry_run else os.environ.get(args.api_key_env)
    cache_path = Path(args.cache)
    cache = {} if args.dry_run else load_cache(cache_path)

    total_chunks = 0
    total_calls = 0
    for source in files:
        chunks, calls = translate_file(
            source,
            out_dir=Path(args.out_dir),
            in_place=args.in_place,
            backup=args.backup,
            cache=cache,
            cache_path=cache_path,
            args=args,
            api_key=api_key,
        )
        total_chunks += chunks
        total_calls += calls

    if args.dry_run:
        print(f"Dry run complete: {len(files)} file(s), {total_chunks} translatable chunk(s).")
    else:
        save_cache(cache_path, cache)
        print(f"Done: {len(files)} file(s), {total_chunks} chunk(s), {total_calls} API call(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
