# quantlab

S&P 500 cross-sectional research framework with rigorous statistics, an honest backtester,
and an Alpaca paper-trading job. Built from the plan in `docs/PLAN.md`.

The point of this code base is not a "winning" strategy. It is a pipeline in which every claim
about a strategy is backed by a test that guards against the standard ways backtests lie:
lookahead, survivorship, multiple testing, cost underestimation, and regime luck.

## Setup

```bash
uv sync --extra dev                 # creates .venv with every dependency
cp .env.example .env                # add ALPACA_API_KEY / ALPACA_SECRET_KEY (paper), optional TIINGO_TOKEN
uv run pytest -q                    # 68 tests: calendar, adjustment, engine invariants, no-lookahead, stats, CV, HMM, Bayes, safety
```

## Data

```bash
uv run quantlab universe build                 # Wikipedia constituents + historical changes -> point-in-time membership
uv run quantlab data update --workers 3        # source chain alpaca -> tiingo -> yfinance, parquet cache, incremental
uv run quantlab data check                     # quality flags, adjustment cross-check, PIT coverage by year
uv run quantlab data snapshot                  # curated panel under data/curated/<ts>, symlinked as latest
uv run quantlab features build                 # every registered feature, cached per snapshot
```

Data facts you should know:

- Yahoo's chart API blocks bare requests from this host; `yfinance` (browser impersonation) works.
  Alpaca is the configured primary once keys exist; it is skipped when they don't.
- Point-in-time price coverage: ~63% of true members in 2005, 82% in 2016, 99% today. Delisted
  names are missing from Yahoo. A free Tiingo token fills most of them. Every tearsheet reports
  the survivorship comparison (PIT membership vs today's constituents) so the bias is visible.
- Yahoo's "unadjusted" close is split-adjusted; Alpaca raw and Tiingo are true raw. Each source
  carries a `split_adjusted` flag and `data/adjust.py` computes our own total-return factor,
  cross-checked against the vendor's adjusted close (discrepancies > 0.5% are flagged).

## Research

```bash
uv run quantlab backtest configs/strategies/momentum_12_1.yaml     # fast: engine only
uv run quantlab run configs/strategies/meanrev_5d.yaml             # full: stats + registry + tearsheet
uv run quantlab stats permute <run_id> --n 500                     # permutation test through the full pipeline
uv run quantlab stats leak-canary <run_id>                         # peeking must inflate Sharpe
uv run quantlab stats param-grid <config> --grid '{"portfolio.max_pos": [0.01,0.02,0.05]}'
uv run quantlab ml train configs/strategies/ml_lgbm.yaml           # purged walk-forward LightGBM
uv run quantlab ml cpcv configs/strategies/ml_lgbm.yaml            # 15-path combinatorial purged CV
uv run quantlab ml shuffle-test configs/strategies/ml_lgbm.yaml    # shuffled labels: OOS IC must be ~0
uv run quantlab registry list                                      # every trial, counted for DSR
```

Tearsheets land in `reports/<run_id>/tearsheet.html`. Each one contains: equity vs SPY, drawdown,
rolling Sharpe, exposure; Lo SE, PSR, DSR (trial count read from the registry, never typed),
MinTRL, stationary-bootstrap CIs; Bayesian P(SR>0) (normal and Student-t); rank-IC series and
decay with Newey-West t; Fama-MacBeth with controls; decile spread and monotonicity; cost
sensitivity with breakeven; regime-conditional table; survivorship comparison.

## Conventions that keep the backtest honest

| Rule | Where it is enforced |
|---|---|
| A feature at `t` uses data `<= t`; never `.shift(-k)` | `features/base.py` contract, `tests/test_no_lookahead.py` over the registry, source scan |
| Only the engine applies the execution lag (decide at close `t`, fill at open `t+1`) | `backtest/engine.py` |
| Costs: commission + half spread + square-root impact + borrow | `portfolio/costs.py`; breakeven on every tearsheet |
| HMM probabilities are filtered, not smoothed | hand-rolled forward filter in `regime/hmm.py`, `tests/test_hmm_filter.py` |
| Every trial is registered; DSR deflates by distinct configs in the family | `registry/runs.py`, `registry/runner.py` |
| Purge + embargo on overlapping labels | `stats/cv.py`, `tests/test_cv.py` |
| Raw prices for filters and sizing, adjusted prices for returns | `data/quality.py`, `data/adjust.py` |

## Bayesian components (`bayes/`)

- `sharpe_posterior.py`: Student-t likelihood via numpyro NUTS plus a conjugate normal fast path.
  The t-location is a robust center; under skew it can disagree with the sample mean, so both are shown.
- `hierarchical.py`: partial pooling of per-stock signal betas toward sector and global means.
- `kalman.py`: random-walk Kalman filter for time-varying betas and a dynamic Fama-MacBeth level.
- `changepoint.py`: Bayesian online changepoint detection; `P(run length <= 10)` is the live alert.
- `black_litterman.py`: reverse-optimized prior with Ledoit-Wolf covariance, signal-implied views.

## Paper trading

```bash
uv run quantlab live run --dry-run       # writes reports/live/<date>/orders_dryrun.json, submits nothing
touch data/KILL                          # kill switch: the job exits immediately while this file exists
uv run quantlab live run --live          # real paper submission (needs keys; still subject to every rail)
uv run quantlab live post-trade          # fills, slippage, reconciliation.md, BOCPD alert on live P&L
```

Safety rails in `live/safety.py` abort before any submission: gross notional, per-position %,
turnover, order count, per-order notional, universe membership, price staleness, account status.
`data/live.sqlite` journals submissions (idempotency: one live submission per session) and daily equity.

Suggested cron (ET): `15 8 * * 1-5 quantlab live run` and `0 10 * * 1-5 quantlab live post-trade`.

## Results so far (net of costs, 2005-2026, PIT universe)

| strategy | gross Sharpe | net Sharpe | note |
|---|---|---|---|
| SPY buy-and-hold (sanity) | 0.65 | 0.65 | engine reproduces the index exactly |
| random long-short (sanity) | 0.16 | -1.74 | costs dominate at 134% weekly turnover |
| momentum 12-1, decile L/S, monthly | 0.00 | -0.05 | IC 0.001; 2009 momentum crash |
| 5-day reversal, rank L/S, weekly | 0.68 | 0.20 | costs eat 70% of the edge; PSR 0.82, DSR 0.81 (6 trials), bootstrap CI [-0.18, 0.58], breakeven at 1.4x costs |
| 5-day reversal + GARCH vol target + HMM regime overlay | 0.66 | -0.04 | overlay lowers exposure but not turnover, so costs win |
| 1-month reversal via Black-Litterman weights, monthly | 0.10 | -0.06 | IC 0.003: the 1-month signal has no power here |
| LightGBM on 14 features, purged walk-forward, weekly | 0.53 | 0.03 | fold IC 0.014, 66% of folds > 0; shuffled-label IC -0.001 (t = -0.96) |
| same model, 15-split CPCV (5 paths) | | 0.11-0.20 | mean 0.15, sd 0.04; trained on more data than walk-forward, so optimistic |

Robustness on the 5-day reversal strategy (`reports/logs/robustness_meanrev.log`):

- Permutation test (200 within-date shuffles through the full pipeline): p = 0.005. Note the null
  carries the same 130% turnover and costs, so its mean Sharpe is -1.9; the test shows the signal
  beats random, not that it beats zero after costs (the bootstrap CI answers that: it includes 0).
- Lookahead canary: peeking one day ahead moves Sharpe from +0.20 to -3.5 (a reversal signal flips
  sign when it sees tomorrow's return), so the honest run is not using future information.
- Rebalance-day offsets: Sharpe by decision weekday = Fri +0.20, Mon -0.16, Tue -0.04, Wed +0.23,
  Thu +0.20. The result depends on the weekday. Treat the headline Sharpe as fragile.
- Parameter grid (feature x max position, 6 cells, all registered as trials): 0.18-0.20; the position
  cap never binds with ~500 names.

Hierarchical shrinkage of per-stock 5-day-reversal betas (last 5 years, `reports/bayes/hierarchical_rev5d_betas.csv`):
unpooled |beta| averages 0.054, shrunk toward sector and global means it averages 0.021 (60% of the
per-name "alpha" is noise). 15.6% of names keep P(beta > 0) > 0.9 (materials and semis lead: ALB, SMCI,
MOS, STLD, ANET); 0.4% are confidently negative (ECHO, TSLA).

These are the honest numbers the framework was built to produce. See the tearsheets for the
full inference (PSR, DSR, bootstrap CI, permutation p) before drawing conclusions.
