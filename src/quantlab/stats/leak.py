"""Lookahead canary: recompute the strategy with features shifted one session INTO THE FUTURE
(deliberate peeking). If the Sharpe does not jump, either the signal carries no information or
the pipeline is already leaking. Reported, never silently passed."""

from __future__ import annotations

from quantlab.registry.experiment import run_pipeline
from quantlab.signals.base import make_signal
from quantlab.stats.sharpe import sharpe


def lookahead_canary(cfg, root, panel, features) -> dict:
    base = sharpe(run_pipeline(cfg, root, panel=panel, features=features).result.returns)
    peek = {k: v.shift(-1) for k, v in features.items()}  # the ONLY intentional negative shift in the code base
    sig = make_signal(cfg, panel, peek)
    cheat = sharpe(run_pipeline(cfg, root, panel=panel, features=features, signal_override=sig).result.returns)
    verdict = "OK: peeking one day ahead inflates Sharpe materially, so the honest run is not using future data" if cheat > base + 0.5 else \
        "SUSPICIOUS: peeking barely changes Sharpe; the signal is either uninformative or something already leaks"
    return {"sharpe_honest": float(base), "sharpe_peeking": float(cheat), "verdict": verdict}
