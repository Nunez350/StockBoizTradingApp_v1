"""Single-file HTML tearsheet (PNG figures embedded as base64)."""

from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).parent / "templates"


def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_tearsheet(ctx: dict, figures: dict[str, Path], out: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    env.filters["pct"] = lambda x, d=1: "n/a" if x is None or x != x else f"{x * 100:.{d}f}%"
    env.filters["num"] = lambda x, d=2: "n/a" if x is None or x != x else f"{x:.{d}f}"
    tpl = env.get_template("tearsheet.html.j2")
    html = tpl.render(**ctx, figs={k: _b64(p) for k, p in figures.items() if p.exists()})
    out.write_text(html)
    return out
