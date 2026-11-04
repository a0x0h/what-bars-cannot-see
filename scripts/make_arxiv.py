"""Assemble a self-contained arXiv source bundle from paper/.

    python paper/make_tables.py
    cd paper && latexmk -pdf main.tex && cd ..
    python scripts/make_arxiv.py

Copies main.tex, the section files, the generated tables, every figure the paper includes (flattened
into figures/) and the compiled bibliography main.bbl (arXiv does not run BibTeX), rewrites the
graphics path, test-compiles the bundle in isolation with pdflatex and writes paper/arxiv_source.zip.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "arxiv_build"
FIG_DIRS = [PAPER / "figures", ROOT / "results" / "BTCUSDT_1H" / "figures", ROOT / "results" / "ETHUSDT_1H" / "figures"]


def main():
    if not (PAPER / "main.bbl").exists():
        sys.exit("paper/main.bbl missing: compile the paper first (latexmk -pdf main.tex)")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "tables").mkdir(parents=True)
    (BUILD / "figures").mkdir()

    tex_files = ["main.tex"] + sorted(p.name for p in PAPER.glob("sec_*.tex"))
    for name in tex_files:
        text = (PAPER / name).read_text(encoding="utf-8")
        if name == "main.tex":
            text = re.sub(r"\\graphicspath\{.*\}", r"\\graphicspath{{figures/}}", text)
        (BUILD / name).write_text(text, encoding="utf-8")
    for t in (PAPER / "tables").glob("*.tex"):
        shutil.copy(t, BUILD / "tables" / t.name)
    shutil.copy(PAPER / "main.bbl", BUILD / "main.bbl")

    figs = set()
    for name in tex_files:
        figs |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", (BUILD / name).read_text(encoding="utf-8")))
    for f in sorted(figs):
        src = next((d / f for d in FIG_DIRS if (d / f).exists()), None)
        if src is None:
            sys.exit(f"figure not found: {f}")
        shutil.copy(src, BUILD / "figures" / Path(f).name)
    print(f"bundle: {len(tex_files)} tex, {len(list((BUILD / 'tables').glob('*.tex')))} tables, {len(figs)} figures")

    # Test-compile in isolation, the way arXiv does (pdflatex, no bibtex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=BUILD,
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            sys.exit("isolated compile failed:\n" + r.stdout[-3000:])
    log = (BUILD / "main.log").read_text(errors="replace")
    problems = [l for l in log.splitlines() if "undefined" in l.lower() or "Citation" in l and "undefined" in l]
    print("isolated compile ok;", "warnings: " + "; ".join(problems[:5]) if problems else "no undefined references")

    out = PAPER / "arxiv_source.zip"
    keep = {".tex", ".bbl", ".png"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUILD.rglob("*")):
            if p.is_file() and p.suffix in keep:
                z.write(p, p.relative_to(BUILD).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
