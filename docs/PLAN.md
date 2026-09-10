# Plan: `quantlab` — S&P 500 quant research framework + paper trading

Project root: `/mnt/main_workspace/stocks` (empty, VERIFIED). Package name `quantlab`, src layout, managed with `uv`.

## 1. Context

The user wants a stock trading strategy built on **strong statistical modeling and honest backtesting**, taken through to **paper trading**. Decisions made via AskUserQuestion:

- **Universe / horizon:** S&P 500, daily bars, multi-day holds (weekly/monthly rebalance).
- **Strategy families:** cross-sectional momentum + short-term mean reversion; ML factor model (LightGBM + small torch MLP); GARCH / HMM regime overlays; **Bayesian methods** (user asked "what other statistical modeling can we leverage, bayesian?" — answered in §7).
- **Data:** user first chose free yfinance. **VERIFIED problem:** from this host Yahoo returns HTTP 429 even with a browser User-Agent, and stooq serves a JS challenge page instead of CSV. User then confirmed: **Alpaca free market data as primary**, Tiingo token optional later for pre-2016 history, yfinance best-effort only (§3).
- **Goal:** framework → one validated strategy report → Alpaca paper trading daily job.

The core deliverable is not a "winning" strategy; it is a framework in which every claim about a strategy is backed by a test that guards against the standard ways backtests lie (lookahead, survivorship, multiple testing, cost underestimation, regime luck).

## 2. Verified environment (2026-09-09)

| Item | Status |
|---|---|
| Python / tooling | 3.12.3, `uv 0.9.11` |
| Installed | numpy 2.4, pandas **3.0.1**, scipy 1.17, scikit-learn 1.8, torch 2.9+cu128, requests |
| Hardware | 24 cores, 188 GB RAM, 2× RTX 5060 Ti 16 GB, 2.4 TB free |
| Network | PyPI 200 · Wikipedia 200 · Yahoo **429** · stooq **JS challenge** · Alpaca paper + data APIs reachable (401 = need keys) · Tiingo reachable (403 = need token) |
| Credentials | none in env or `~/.env` → user must create `.env` with `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (and optional `TIINGO_TOKEN`) |
| Latest PyPI | statsmodels 0.15, arch 8.0, hmmlearn 0.3.3, lightgbm 4.7, yfinance 1.7, alpaca-py 0.44, pyarrow 25, matplotlib 3.11, numpyro 0.21, jax 0.11, exchange-calendars 4.13 |

Risk (INFERRED, medium confidence): pandas 3.0 copy-on-write and default `str` dtype may break older code paths in arch/statsmodels/hmmlearn. Guard: M0 compat smoke test; single `compat.py` shim point.

## 3. Data layer

**Source chain (per symbol, first success wins), normalized to one raw schema** `date, open, high, low, close, volume, split_ratio, dividend`:

1. `AlpacaSource` — free Basic plan daily bars (`adjustment='raw'`, `feed='iex'` on free tier; `sip` if the account allows) + corporate actions. History back to ~2016. The user needs Alpaca keys for paper trading anyway, so this costs nothing extra.
2. `TiingoSource` — free token; raw + adjClose + split/div back to the 1990s, includes many delisted names. Free-tier rate limits make the backfill a one-time job; the cache makes that fine.
3. `YFinanceSource` — best effort only (yfinance 1.7 uses curl_cffi browser impersonation, which *may* bypass the 429; unverified). Backoff on 429, never blocks the pipeline.

**Cache:** `data/raw/{source}/{SYMBOL}.parquet` with metadata; `update_incremental` re-fetches from `last_date − 5 sessions` and merges. Second run = no-op.

**Adjustment:** compute our own backward cumulative factor from splits/dividends → `adj_*` for returns/features; **raw** prices for min-price filters, share sizing, live fills. Cross-check vs Tiingo adjClose; flag >0.5% discrepancies.

**Universe (survivorship mitigation):** scrape Wikipedia current table + "Selected changes" table; reconstruct point-in-time (PIT) membership backward. Limitations documented (changes table sparse pre-2000; removed tickers may lack prices). Guards: monthly coverage stat; `universe: pit | current` both run and shown side-by-side in the tearsheet so survivorship premium is quantified; engine liquidates at last close on delisting.

**Calendar:** `exchange-calendars` XNYS; Alpaca UTC → NY session dates; assert alignment.

**Quality (`data/quality.py`):** dup dates, non-positive prices, zero volume, |ret|>50% w/o split, stale closes, gaps, OHLC sanity, adj discrepancies → `quality.json` + `tradable` mask (member ∧ raw close > $5 ∧ 20d ADV > $5M ∧ no critical flag). Snapshot id = sha256 of curated parquet; every run records it.

## 4. Package layout

```
pyproject.toml  .env.example  .gitignore  configs/{base.yaml, strategies/*.yaml, overlays/*.yaml, paper.yaml}
data/ (gitignored): raw/ curated/{snapshot}/ runs/{run_id}/ registry.sqlite live.sqlite KILL
reports/{run_id}/tearsheet.html
src/quantlab/
  config.py (pydantic), calendar.py, seeds.py, compat.py, cli.py
  data/       sources.py cache.py universe.py adjust.py quality.py panel.py snapshot.py
  features/   base.py (contract + registry + per-date scalers) price.py registry.py
  signals/    base.py momentum.py meanrev.py ml.py combine.py
  portfolio/  construct.py risk.py costs.py
  backtest/   engine.py result.py
  stats/      newey_west.py ic.py fama_macbeth.py deciles.py cv.py bootstrap.py sharpe.py
              permutation.py sensitivity.py regime.py drawdown.py leak.py
  ml/         dataset.py models.py train.py predict.py
  regime/     garch.py hmm.py overlay.py
  bayes/      sharpe_posterior.py hierarchical.py kalman.py changepoint.py black_litterman.py
  registry/   runs.py experiment.py
  report/     plots.py tearsheet.py templates/tearsheet.html.j2
  live/       broker.py targets.py reconcile.py safety.py daily_job.py state.py
tests/  test_no_lookahead.py test_engine.py test_stats_synthetic.py test_cv.py test_hmm_filter.py
        test_bayes.py test_live_safety.py test_universe.py test_adjust.py test_cache.py
```

**Dependencies (runtime):** numpy, pandas>=3, scipy, scikit-learn, pyarrow, pydantic>=2, pyyaml, python-dotenv, requests, exchange-calendars, alpaca-py, yfinance, lightgbm, torch>=2.9, arch, statsmodels, hmmlearn, numpyro + jax (CPU; chosen over pymc: lighter, faster NUTS, no compile step), matplotlib, jinja2. **Dev:** pytest, pytest-xdist, hypothesis, ruff. Not added: vectorbt, polars, duckdb (pandas is enough at 500 × 25y daily).

## 5. Features, signals, and the no-lookahead contract

Contract (in `features/base.py` docstring, enforced by test):
- Feature at row `t` uses only data with session date ≤ t. No `.shift(-k)` anywhere in features. Forward returns exist only in `ml/dataset.py` and `stats/*`, named `fwd_*`.
- Cross-sectional scaling is per-date; time-series scaling is expanding/rolling ending at t.
- **Only the engine applies the execution lag.** Strategies emit scores at decision date t; the engine trades at open t+1.

Enforcement: `tests/test_no_lookahead.py` iterates every registered feature (registration decorator → cannot add a feature without coverage), computes on full panel vs `panel.slice(end=t)`, asserts equality at t. `stats/leak.py` canary: recompute features on a panel shifted −1 day (peeking) and assert Sharpe inflates materially.

Initial features: mom_12_1, mom_6_1, 1m reversal, 5d reversal, rvol 20/60, idio vol (vs SPY, 60d), beta_252, log ADV, Amihud, max 5d return, 52w-high proximity, 60d skew. Signals: momentum (rank, optional vol-scaled), mean reversion (−rank(5d), liquid names only), combiner (rank-average / IC-weighted on trailing window only).

## 6. Portfolio construction + backtest engine

**Vectorized daily engine with a rebalance-date loop** (not event-driven): daily bars + weekly/monthly rebalance have no intraday events, and <1 s/run makes 500-permutation tests and CPCV feasible.

Execution: decide at close t, fill at open t+1 at `open × (1 ± half_spread ± impact)`; P&L marked open-to-open; weights drift between rebalances. Cost model: commission + half-spread bps + square-root impact `coef · σ_daily · sqrt(notional / ADV$)` (coef 0.5 default) + borrow bps on shorts. Constraints: max position %, dollar-neutral or beta-neutral (explicit SPY hedge from betas estimated to t), long-only option. Vol targeting from strictly-lagged strategy returns (GARCH forecast when overlay enabled).

`BacktestResult`: daily net/gross returns, equity, daily positions, trades, turnover, exposures, metadata (config hash, snapshot id, git sha, seed). Parquet round-trip.

Invariants (`tests/test_engine.py`): zero weights → zero return; SPY buy-and-hold matches adj-close total return to 1e-8 at zero cost; more cost never increases net return; symbol-order permutation invariance (hypothesis).

## 7. Statistics module (the core requirement)

| Estimator | Gotcha handled |
|---|---|
| Newey-West mean t-stat (Bartlett) | lags = horizon−1 for overlapping forward returns |
| IC / rank-IC series, summary, decay (h = 1…63) | ≥30 names/date; fwd returns open(t+1)→open(t+1+h) to match execution; split-half IC |
| Fama-MacBeth with controls (beta, log ADV as size proxy, rvol) | per-date winsorize 1/99 + standardize; NW on gammas |
| Decile portfolios, D10−D1 spread, monotonicity test | bootstrap p on Spearman of decile means |
| `PurgedKFold`, `CombinatorialPurgedCV` (6 groups, 2 test → 15 paths), walk-forward | purge by label-interval overlap + embargo; all rows of a date in the same fold |
| Stationary bootstrap CI (Sharpe, CAGR, max DD) | Politis-Romano; block length auto; DD CI flagged as downward-biased |
| Sharpe SE (Lo), **PSR, DSR, expected max SR, MinTRL, effective trials** | `n_trials` read from the run registry, never hand-typed; effective trials via clustering of trial-return correlations |
| Permutation test (within-date score shuffle, and time shuffle), n=500, full engine incl. costs | joblib parallel |
| Cost sensitivity curve + breakeven bps; parameter-stability heatmap; rebalance-offset dispersion | every grid cell logged as a trial |
| Regime-conditional performance | labels must be filtered/lagged (same no-lookahead test) |

Synthetic tests: plant `fwd_ret = ρ·signal + noise`, recover ρ via IC and FM within CI; PSR/DSR match Bailey & López de Prado (2014) worked numbers.

## 8. Bayesian components (answer to "bayesian?") — build in this order

1. **Bayesian Sharpe / P(SR > 0)** — Student-t likelihood on daily returns, NUTS via numpyro (~2 s), plus conjugate Normal-Inverse-Gamma fast path. Reports P(SR>0), P(SR>0.5), posterior predictive. On every tearsheet next to PSR/DSR. Robust to fat tails where the frequentist t-test is not.
2. **Hierarchical shrinkage of per-stock signal betas** — partial pooling toward sector and global means (numpyro). Shows how much per-name "alpha" survives shrinkage; yields shrunk expected returns for construction.
3. **Kalman filter** (numpy, ~30 lines) — random-walk time-varying market beta (replaces rolling OLS for the hedge) and dynamic Fama-MacBeth coefficient to detect signal decay.
4. **Bayesian online changepoint detection** (Adams & MacKay, conjugate NIG, numpy) — on strategy P&L and market vol; live alert for structural breaks; alternative regime overlay.
5. **Black-Litterman** (numpy) — prior from reverse optimization with Ledoit-Wolf covariance; views = signal-implied returns with confidence from trailing IC; posterior → mean-variance weights under the same constraints. Exposed as `method='bl'` in `scores_to_weights`.
6. Later: Bayesian model averaging of strategies weighted by P(SR>0).

## 9. Regime / vol overlays

- **GARCH(1,1)-t** on SPY via `arch` (returns ×100, rescaled back), refit monthly expanding, 1-step variance forecast daily. Per-stock vol via EWMA.
- **HMM** (`hmmlearn` GaussianHMM, 2–3 states on [SPY ret, 20d rvol, cross-sectional dispersion]) refit monthly. **Critical:** `predict_proba` is forward-backward (uses the future). Hand-roll the forward filter on fitted params; relabel states by sorted variance. Test: appending future rows doesn't change earlier filtered probs.
- `VolTargetOverlay`, `RegimeOverlay` (multiplier = 1 − cut·P(high-vol)), `CompositeOverlay`. Engine consumes any overlay as a lagged multiplier series.

## 10. ML pipeline

Long (date, symbol) dataset with lagged features; label = h-day forward open-to-open return, cross-sectionally z-scored per date; `t0/t1` for purging. Walk-forward: expanding train, retrain quarterly, embargo = h; inner PurgedKFold for hyperparameters on train slice only; CPCV once per config → 15-path OOS Sharpe distribution. Every hyperparameter config logged as a trial (`kind='ml_hp'`). Models: LightGBM (`deterministic=True`, fixed seed, 24 threads) and TorchMLP (2×64, dropout, early stop on purged val fold, CUDA:0, deterministic flags). Predictions → same `Signal` → same constructor → same engine, so all stats apply unchanged. Leakage test: shuffled-label OOS IC |t| < 2.

## 11. Paper trading (`live/`)

`daily_job.main(dry_run)` at 08:15 ET on sessions:
1. Exit if `data/KILL` exists; exit if today's orders already recorded (idempotency).
2. Incremental data update via Alpaca; quality check on the tail; abort if >2% of universe stale.
3. Compute signal at yesterday's close → overlays → target weights → target shares (raw close × account equity).
4. Diff vs broker positions → orders; drop trades < $200 notional.
5. `SafetyRails.check`: max gross notional, max position %, max daily turnover %, max order count, price staleness, symbol-in-universe, per-order cap. Any violation → abort before any submission.
6. `dry_run=True` (default) writes `reports/live/{date}/orders_dryrun.json`; else submit market-on-open (`time_in_force='opg'`) before 09:28 ET; record in `live.sqlite`.
7. 10:00 ET `post_trade()`: fills, slippage vs prior close, positions vs targets → `reconciliation.md`; BOCPD on live P&L for break alerts.

`configs/paper.yaml`: caps above, `dry_run: true`, `strategy_ref` = run_id of the approved backtest so live config equals a registered trial.

## 12. Reproducibility, registry, tearsheet

- `seeds.set_all_seeds`; seed, config hash, snapshot id, git sha recorded per run. `Experiment.run` refuses a dirty tree unless `--allow-dirty` (project will be `git init`ed in M0).
- `RunRegistry` (SQLite `trials`: run_id, family, kind ∈ {trial, cv_fold, permutation, ml_hp, sanity}, config, snapshot, git_sha, seed, n_obs, sharpe, skew, kurt, metrics). Daily returns per run saved so effective-trials correlation is computable. Permutation runs are logged but excluded from the DSR trial count.
- Tearsheet (matplotlib + jinja2 HTML): header (config, snapshot, sha, seed, n_trials); equity vs SPY; drawdown; rolling Sharpe; IC series + decay; deciles; FM table; turnover + cost curve + breakeven; param heatmap; regime table; bootstrap CI table; PSR/DSR/MinTRL; Bayesian P(SR>0) + posterior plot; permutation p; PIT-vs-current survivorship comparison; leak-canary result.

## 13. Milestones (each ends with a demonstrable check)

| # | Scope | Verification |
|---|---|---|
| M0 | `git init`, `uv init`, pyproject, src layout, config/pydantic, calendar, seeds, CLI skeleton, pandas-3 compat smoke test | `uv sync && uv run pytest -q` green; `quantlab calendar 2024-01-02 2024-01-10` → 6 sessions |
| M1 | Wikipedia scrape, PIT membership | `quantlab universe build` → ~503 members + yearly coverage table; `membership.parquet` exists |
| M2 | Source chain, cache, adjustment, quality, snapshot | `quantlab data update --symbols SPY,AAPL,BRK.B` then `data check` → 0 critical; 2nd update → "0 fetched"; then full backfill |
| M3 | Feature registry + price features | `pytest tests/test_no_lookahead.py` passes for every feature |
| M4 | Constructor, cost model, engine, result | `spy_hold.yaml` reproduces SPY TR within 1e-6 at zero cost; `random.yaml` net Sharpe ≈ 0, negative after costs |
| M5 | NW, IC, deciles, FM, PSR/DSR, bootstrap, CV splitters | synthetic tests recover planted ρ; DSR matches paper; purge test shows no overlap |
| M6 | Registry, Experiment, tearsheet | `quantlab run configs/strategies/momentum_12_1.yaml` → tearsheet; registry shows `n_trials=1` |
| M7 | Momentum + mean reversion with full rigor (cost curve, param grid, offset dispersion, PIT vs current) | both tearsheets; registry grows by exactly grid size; DSR uses that count |
| M8 | ML dataset, purged walk-forward, CPCV, LightGBM, TorchMLP | per-fold IC + 15-path CPCV Sharpe; shuffled-label IC \|t\|<2; GPU log line |
| M9 | GARCH, hand-rolled HMM filter, overlays, regime stats | `test_hmm_filter.py` passes; overlay tearsheet shows regime table |
| M10 | Bayesian: Sharpe posterior, hierarchical betas, Kalman, BOCPD, BL | `test_bayes.py`: recover planted params; BOCPD flags planted break ≤10 days; `method: bl` backtest runs |
| M11 | Broker wrapper, targets, reconcile, safety, daily job, cron | `quantlab live run --dry-run` writes orders; `touch data/KILL` → exits; with keys, $5k paper submission + `post-trade` → `reconciliation.md` |
| M12 | Permutation test, leak canary, determinism | `stats permute <run> --n 500` → p-value; canary shows inflation under peeking; two identical runs → identical metrics hash |

**Blocking user inputs:** Alpaca paper API keys (needed from M2 for data, M11 for orders); optional Tiingo token for pre-2016 history. Until keys exist, M2 can be built and unit-tested against recorded fixture responses, and yfinance will be tried as best-effort.

## 14. Pitfalls → guard location

| Pitfall | Guard |
|---|---|
| Feature lookahead | `tests/test_no_lookahead.py` over registry; engine-only lag; `stats/leak.py` canary |
| HMM smoothing peeks | hand-rolled forward filter + `test_hmm_filter.py` |
| Adjusted prices in filters | `quality.tradable_mask` uses raw prices |
| Survivorship | PIT membership; coverage stat; PIT-vs-current on tearsheet; delisting liquidation |
| Overlapping returns inflate t | NW lags tied to horizon everywhere |
| Multiple testing | every run in registry; `dsr()` reads counts from registry |
| Rebalance-timing luck | `rebalance_offset_dispersion` |
| Cost underestimation | sqrt impact + spread + borrow; breakeven on tearsheet |
| Full-sample normalization leak | `features/base.py` exposes only per-date / expanding scalers |
| ML purge failures | interval-based purge + embargo; shuffled-label test |
| Non-determinism | `seeds.py`; identical-hash check M12 |
| Ticker canonicalization (BRK.B vs BRK-B), reuse | `universe.canonical_ticker`; membership keyed (symbol, date) |
| Data-source blocking | source chain, Alpaca primary, parquet cache |
| pandas 3 compat | M0 smoke test; `compat.py` |
| Live double-submit / runaway orders | idempotency guard, kill switch, caps, dry-run default |
| Timezones on Alpaca bars | `calendar.py` UTC→NY session assertion |

## 15. Honest expectations

INFERRED (high confidence from the literature): after realistic costs and DSR correction, plain 12-1 momentum and 5-day reversal on S&P 500 names over 2005–2026 will likely show modest, regime-dependent Sharpe (roughly 0.3–0.8 gross, lower net). The framework is designed to *report* that honestly rather than tune until it looks good. The value is in the validated pipeline; strategy quality is an empirical outcome, not a promise.
