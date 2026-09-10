"""Tearsheet figures (matplotlib, light surface). Palette and mark specs follow the
project's data-viz conventions: thin marks, hairline grid, categorical hues in fixed
order (blue, orange, aqua), sequential blue ramp, blue<->red diverging, text in ink tokens."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, BLUE = "#e34948", "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_rb", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round", "font.family": "sans-serif",
})


def _fig(w=9, h=3.2):
    return plt.subplots(figsize=(w, h), dpi=110)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def equity_curve(returns: pd.Series, bench: pd.Series | None, path, title="Cumulative return (log scale)"):
    fig, ax = _fig()
    eq = (1 + returns.fillna(0)).cumprod()
    ax.plot(eq.index, eq.values, color=S1, label="strategy")
    if bench is not None:
        b = (1 + bench.reindex(eq.index).fillna(0)).cumprod()
        ax.plot(b.index, b.values, color=S2, label="SPY total return", linewidth=1.5)
        ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.annotate(f"{eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, va="center")
    _save(fig, path)


def drawdown(returns: pd.Series, path):
    fig, ax = _fig(h=2.4)
    eq = (1 + returns.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.12, linewidth=0)
    ax.plot(dd.index, dd.values, color=RED, linewidth=1.5)
    i = dd.idxmin()
    ax.annotate(f"{dd.min():.1%}", (i, dd.min()), xytext=(6, -2), textcoords="offset points", color=INK2, va="top")
    ax.set_title("Drawdown")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, path)


def rolling_sharpe(returns: pd.Series, path, window=252):
    fig, ax = _fig(h=2.6)
    r = returns.dropna()
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(rs.index, rs.values, color=S1)
    ax.set_title(f"Rolling {window}-day Sharpe")
    _save(fig, path)


def ic_series_plot(ic: pd.Series, path, window=12):
    fig, ax = _fig(h=2.8)
    ic = ic.dropna()
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.bar(ic.index, ic.values, width=max(1, (ic.index[1] - ic.index[0]).days * 0.8) if len(ic) > 1 else 5,
           color=S1, alpha=0.35, linewidth=0)
    m = ic.rolling(window, min_periods=max(3, window // 2)).mean()
    ax.plot(m.index, m.values, color=S1, label=f"{window}-period mean")
    ax.legend(loc="upper left")
    ax.set_title(f"Rank IC per decision date (mean {ic.mean():.3f})")
    _save(fig, path)


def ic_decay_plot(decay: pd.DataFrame, path):
    fig, ax = _fig(w=5, h=2.8)
    x = np.arange(len(decay))
    ax.bar(x, decay["mean_ic"].values, color=S1, width=0.6, linewidth=0)
    for xi, (h, row) in zip(x, decay.iterrows()):
        ax.annotate(f"t={row['t_nw']:.1f}", (xi, row["mean_ic"]), xytext=(0, 3 if row["mean_ic"] >= 0 else -10),
                    textcoords="offset points", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, [str(h) for h in decay.index])
    ax.set_xlabel("horizon (sessions)")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("IC decay by holding horizon")
    ax.grid(axis="y")
    _save(fig, path)


def decile_bars(decile_means: dict, path, periods_per_year: float):
    fig, ax = _fig(w=5, h=2.8)
    ks = sorted(decile_means)
    vals = np.array([decile_means[k] for k in ks]) * periods_per_year
    ax.bar(np.arange(len(ks)), vals, color=S1, width=0.6, linewidth=0)
    ax.set_xticks(np.arange(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("signal decile (10 = highest score)")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Annualized mean forward return by decile")
    ax.grid(axis="y")
    _save(fig, path)


def cost_curve(table: pd.DataFrame, path, breakeven: float | None):
    fig, ax = _fig(w=5, h=2.8)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(table["multiplier"], table["sharpe"], color=S1, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if breakeven is not None and np.isfinite(breakeven):
        ax.axvline(breakeven, color=MUTED, linewidth=1)
        ax.annotate(f"breakeven ×{breakeven:.2f}", (breakeven, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("cost multiplier (1 = base assumptions)")
    ax.set_title("Net Sharpe vs transaction-cost multiplier")
    _save(fig, path)


def param_heatmap(grid: pd.DataFrame, path, title="Sharpe across parameter grid"):
    fig, ax = _fig(w=5, h=3.2)
    v = grid.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1
    im = ax.imshow(v, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index])
    ax.set_xlabel(grid.columns.name or "")
    ax.set_ylabel(grid.index.name or "")
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK if abs(v[i, j]) < 0.6 * lim else SURFACE)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title)
    _save(fig, path)


def histogram(samples: np.ndarray, path, title: str, vline: float | None = None, xlabel: str = ""):
    fig, ax = _fig(w=5, h=2.6)
    ax.hist(samples, bins=50, color=S1, alpha=0.8, linewidth=0)
    if vline is not None:
        ax.axvline(vline, color=INK2, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(axis="y")
    _save(fig, path)


def exposure_plot(exposure: pd.DataFrame, path):
    fig, ax = _fig(h=2.4)
    ax.plot(exposure.index, exposure["gross"].values, color=S1, label="gross")
    ax.plot(exposure.index, exposure["net"].values, color=S2, label="net", linewidth=1.5)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.legend(loc="upper left")
    ax.set_title("Exposure (fraction of NAV)")
    _save(fig, path)


def offsets_plot(table: pd.DataFrame, path):
    fig, ax = _fig(w=5, h=2.6)
    ax.bar(np.arange(len(table)), table["sharpe"].values, color=S1, width=0.6, linewidth=0)
    ax.set_xticks(np.arange(len(table)), [str(x) for x in table["offset"]])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("rebalance offset (sessions)")
    ax.set_title("Net Sharpe by rebalance-day offset")
    ax.grid(axis="y")
    _save(fig, path)
