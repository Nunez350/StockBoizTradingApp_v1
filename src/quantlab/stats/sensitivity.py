"""Robustness analyses that re-run the full pipeline: cost curves, parameter grids, rebalance offsets."""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

from quantlab.stats.sharpe import sharpe


def cost_sensitivity(run_fn, multipliers=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> tuple[pd.DataFrame, float | None]:
    """``run_fn(cost_scale) -> returns``. Returns the table and the breakeven multiplier (Sharpe = 0)."""
    rows = []
    for m in multipliers:
        r = run_fn(m)
        rows.append({"multiplier": m, "sharpe": sharpe(r), "cagr": float((1 + r).prod() ** (252 / len(r)) - 1)})
    t = pd.DataFrame(rows)
    be = None
    s = t["sharpe"].to_numpy()
    for i in range(1, len(s)):
        if s[i - 1] > 0 >= s[i]:
            x0, x1 = t["multiplier"].iloc[i - 1], t["multiplier"].iloc[i]
            be = float(x0 + (x1 - x0) * s[i - 1] / (s[i - 1] - s[i]))
            break
    if be is None and s[-1] > 0:
        be = np.inf
    return t, be


def set_param(cfg, dotted: str, value):
    """Return a copy of ``cfg`` with ``a.b.c`` set to value (supports dict leaves like signal.params.feature)."""
    d = cfg.model_dump()
    node = d
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return type(cfg).model_validate(d)


def param_stability(cfg, run_fn, grid: dict[str, list], log=None) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate every combination of a 1- or 2-parameter grid. ``run_fn(cfg) -> returns``.
    Returns a 2-D Sharpe table (rows = first param) plus a list of per-cell records for the registry."""
    keys = list(grid)
    if len(keys) > 2:
        raise ValueError("grid supports at most two parameters")
    records = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        c = cfg
        for k, v in zip(keys, combo):
            c = set_param(c, k, v)
        r = run_fn(c)
        rec = {k: v for k, v in zip(keys, combo)}
        half = len(r) // 2
        rec.update({"sharpe": sharpe(r), "sharpe_h1": sharpe(r.iloc[:half]), "sharpe_h2": sharpe(r.iloc[half:]),
                    "returns": r, "config": c})
        records.append(rec)
        if log:
            log(f"  grid {rec[keys[0]]}" + (f" x {rec[keys[1]]}" if len(keys) > 1 else "") + f" -> sharpe {rec['sharpe']:.2f}")
    df = pd.DataFrame([{k: r[k] for k in keys} | {"sharpe": r["sharpe"]} for r in records])
    if len(keys) == 1:
        table = df.set_index(keys[0])[["sharpe"]].T
        table.index = ["sharpe"]
        table.columns.name = keys[0]
    else:
        table = df.pivot(index=keys[0], columns=keys[1], values="sharpe")
    return table, records


def rebalance_offset_dispersion(cfg, run_fn, max_offset: int) -> pd.DataFrame:
    """Shift the decision dates by 0..max_offset sessions and re-run. ``run_fn(cfg, offset) -> returns``."""
    rows = []
    for off in range(max_offset + 1):
        r = run_fn(cfg, off)
        rows.append({"offset": off, "sharpe": sharpe(r), "n": len(r)})
    return pd.DataFrame(rows)
