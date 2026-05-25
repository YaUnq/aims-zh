# Chinese Translation Sources

This directory contains the Chinese Quarto translation of the book.

- `index.qmd` and `license.qmd` are translated root-level pages.
- `src/*.qmd` contains translated chapter sources.
- `_book/` is generated output and is intentionally ignored by Git.

Render the Chinese HTML output from the repository root:

```bash
python scripts/render_zh_book.py
python -m http.server 8001 --directory zh/_book
```

The render helper builds a temporary Quarto project from the original book
configuration and overlays these translated sources before rendering.
