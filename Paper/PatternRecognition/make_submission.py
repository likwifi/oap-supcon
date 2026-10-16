#!/usr/bin/env python3
"""Assemble the Editorial Manager upload set in submission/ and PR_submission.zip.

Usage: python make_submission.py BUILD_DIR
BUILD_DIR holds the compiled main.pdf, main.bbl, supplementary.pdf and
cover_letter.pdf (for example from `latexmk -pdf` or `tectonic --keep-intermediates`).
The flattened manuscript.tex inlines the generated tables and embeds the
bibliography, so it compiles with pdfLaTeX and elsarticle.cls alone.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def flatten(build: Path) -> str:
    text = (HERE / "main.tex").read_text()

    def table(match):
        body = (HERE / match.group(1)).read_text()
        return "\n".join(line for line in body.splitlines() if not line.startswith("% Generated"))

    text, count = re.subn(r"\\input\{(tables/[^}]+\.tex)\}", table, text)
    if count == 0:
        raise SystemExit("no generated tables were inlined")
    bbl = (build / "main.bbl").read_text()
    marker = "\\bibliographystyle{elsarticle-num}\n\\bibliography{references}"
    if marker not in text:
        raise SystemExit("bibliography commands not found")
    text = text.replace(marker, "% Bibliography generated with elsarticle-num.bst from references.bib.\n" + bbl.strip())
    if "\\input{" in text or "\\bibliography{" in text:
        raise SystemExit("unresolved external input remains")
    return text


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    build = Path(sys.argv[1]).resolve()
    out = HERE / "submission"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    (out / "manuscript.tex").write_text(flatten(build))
    for name in ("fig1.pdf", "fig2.pdf", "fig3.pdf", "fig4.pdf", "highlights.txt"):
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(build / "main.pdf", out / "manuscript.pdf")
    shutil.copy2(build / "supplementary.pdf", out / "supplementary_material.pdf")
    shutil.copy2(build / "cover_letter.pdf", out / "cover_letter.pdf")
    archive = HERE / "PR_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.iterdir()):
            handle.write(path, path.name)
    print("wrote", out, "and", archive)
    for path in sorted(out.iterdir()):
        print(f"  {path.name:28s} {path.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main()
