"""Feature registry access and the feature-panel builder/cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import quantlab.features.price  # noqa: F401  (registers features)
from quantlab.data.panel import Panel, snapshot_dir
from quantlab.features.base import FEATURES, compute


def list_features() -> list[str]:
    return sorted(FEATURES)


def build_features(panel: Panel, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list_features()
    return {n: compute(n, panel) for n in names}


def to_long(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack to a long frame indexed by (date, symbol) with one column per feature."""
    cols = {n: df.stack(future_stack=True) for n, df in features.items()}
    out = pd.DataFrame(cols)
    out.index.names = ["date", "symbol"]
    return out


def feature_cache_dir(root: Path, snapshot: str = "latest") -> Path:
    return snapshot_dir(root, snapshot) / "features"


def load_or_build(root: Path, panel: Panel, names: list[str] | None = None, snapshot: str = "latest") -> dict[str, pd.DataFrame]:
    d = feature_cache_dir(root, snapshot)
    d.mkdir(parents=True, exist_ok=True)
    names = names or list_features()
    out = {}
    for n in names:
        p = d / f"{n}.parquet"
        if p.exists():
            out[n] = pd.read_parquet(p)
        else:
            out[n] = compute(n, panel)
            out[n].to_parquet(p)
    return out


def cli_features(a) -> None:
    root = Path(a.root)
    if a.action == "list":
        for n in list_features():
            print(n)
        return
    from quantlab.data.panel import load_panel

    panel = load_panel(root, a.snapshot)
    feats = load_or_build(root, panel, snapshot=a.snapshot)
    for n, df in feats.items():
        last = df.iloc[-1]
        print(f"{n:14s} shape={df.shape} non-null(last)={int(last.notna().sum())} mean(last)={last.mean():.4f}")
    print(f"written to {feature_cache_dir(root, a.snapshot)}")
