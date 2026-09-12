"""``quantlab`` command-line interface.

Subcommands import their implementation lazily so that a broken optional dependency
in one area (e.g. jax) does not take the whole CLI down.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_calendar(a):
    from quantlab.calendar import sessions

    s = sessions(a.start, a.end)
    for d in s:
        print(d.date())
    print(f"{len(s)} sessions")


def _cmd_compat(a):
    from quantlab.compat import smoke_test

    res = smoke_test()
    bad = 0
    for k, v in res.items():
        print(f"{k:20s} {v}")
        bad += v != "ok"
    sys.exit(1 if bad else 0)


def _cmd_universe(a):
    from quantlab.data.universe import cli_build

    cli_build(a)


def _cmd_data(a):
    from quantlab.data.cli import cli_data

    cli_data(a)


def _cmd_features(a):
    from quantlab.features.registry import cli_features

    cli_features(a)


def _cmd_backtest(a):
    from quantlab.registry.experiment import cli_backtest

    cli_backtest(a)


def _cmd_run(a):
    from quantlab.registry.experiment import cli_run

    cli_run(a)


def _cmd_registry(a):
    from quantlab.registry.runs import cli_registry

    cli_registry(a)


def _cmd_stats(a):
    from quantlab.stats.cli import cli_stats

    cli_stats(a)


def _cmd_ml(a):
    from quantlab.ml.train import cli_ml

    cli_ml(a)


def _cmd_live(a):
    from quantlab.live.daily_job import cli_live

    cli_live(a)


def _cmd_api(a):
    from quantlab.api.app import cli_api

    cli_api(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calendar", help="print NYSE sessions in a range")
    c.add_argument("start")
    c.add_argument("end")
    c.set_defaults(fn=_cmd_calendar)

    c = sub.add_parser("compat", help="smoke-test fragile dependencies under pandas 3")
    c.set_defaults(fn=_cmd_compat)

    c = sub.add_parser("universe", help="S&P 500 point-in-time membership")
    c.add_argument("action", choices=["build", "show"])
    c.add_argument("--root", default="data")
    c.add_argument("--start", default="1996-01-01")
    c.add_argument("--refresh", action="store_true", help="re-download Wikipedia tables")
    c.set_defaults(fn=_cmd_universe)

    c = sub.add_parser("data", help="download / cache / check price data")
    c.add_argument("action", choices=["update", "check", "snapshot", "status"])
    c.add_argument("--root", default="data")
    c.add_argument("--symbols", default=None, help="comma list; default = universe + benchmark")
    c.add_argument("--start", default="1996-01-01")
    c.add_argument("--sources", default="alpaca,tiingo,yfinance")
    c.add_argument("--workers", type=int, default=4)
    c.set_defaults(fn=_cmd_data)

    c = sub.add_parser("edgar", help="SEC EDGAR fundamentals + earnings dates")
    c.add_argument("action", choices=["fetch", "map"])
    c.add_argument("--root", default="data")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--refresh", action="store_true")
    c.set_defaults(fn=lambda a: __import__("quantlab.data.edgar", fromlist=["cli_edgar"]).cli_edgar(a))

    c = sub.add_parser("features", help="build the feature panel")
    c.add_argument("action", choices=["build", "list"])
    c.add_argument("--root", default="data")
    c.add_argument("--snapshot", default="latest")
    c.set_defaults(fn=_cmd_features)

    c = sub.add_parser("backtest", help="run a backtest without stats/registry (fast)")
    c.add_argument("config")
    c.add_argument("--root", default=None)
    c.set_defaults(fn=_cmd_backtest)

    c = sub.add_parser("run", help="full experiment: backtest + stats + registry + tearsheet")
    c.add_argument("config")
    c.add_argument("--root", default=None)
    c.add_argument("--allow-dirty", action="store_true")
    c.add_argument("--no-report", action="store_true")
    c.add_argument("--quick", action="store_true", help="skip slow stats (bootstrap, permutation, bayes)")
    c.add_argument("--n-perm", type=int, default=200)
    c.add_argument("--n-boot", type=int, default=2000)
    c.set_defaults(fn=_cmd_run)

    c = sub.add_parser("registry", help="inspect the run registry")
    c.add_argument("action", choices=["list", "show", "count"])
    c.add_argument("--root", default="data")
    c.add_argument("--family", default=None)
    c.add_argument("--run-id", default=None)
    c.set_defaults(fn=_cmd_registry)

    c = sub.add_parser("stats", help="standalone statistical tests on a registered run")
    c.add_argument("action", choices=["permute", "leak-canary", "cost-curve", "param-grid", "offsets", "bayes"])
    c.add_argument("target", help="run id or config path depending on action")
    c.add_argument("--root", default="data")
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--grid", default=None, help="JSON dict of param -> list for param-grid")
    c.add_argument("--workers", type=int, default=8)
    c.set_defaults(fn=_cmd_stats)

    c = sub.add_parser("ml", help="train ML factor models with purged walk-forward / CPCV")
    c.add_argument("action", choices=["train", "cpcv", "shuffle-test", "meta-train"])
    c.add_argument("config")
    c.add_argument("--root", default=None)
    c.set_defaults(fn=_cmd_ml)

    c = sub.add_parser("live", help="paper-trading daily job")
    c.add_argument("action", choices=["run", "post-trade", "status", "cancel-all"])
    c.add_argument("--config", default="configs/paper.yaml")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--live", action="store_true", help="actually submit orders (overrides config dry_run)")
    c.set_defaults(fn=_cmd_live)

    c = sub.add_parser("api", help="serve the StockBoiz web UI + JSON API")
    c.add_argument("action", choices=["serve"])
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--root", default="data")
    c.add_argument("--reports", default="reports")
    c.add_argument("--live-config", default="configs/paper.yaml")
    c.add_argument("--strategies", default="configs/strategies")
    c.add_argument("--web-dist", default="web/dist", help="built front end (npm run build in web/)")
    c.add_argument("--snapshot", default="latest")
    c.set_defaults(fn=_cmd_api)
    return p


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    p = build_parser()
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    main()
