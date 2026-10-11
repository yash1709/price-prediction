# NSE Stock Price Prediction

A research and forecasting application for NSE-listed stocks: it fetches end-of-day prices, engineers
leakage-safe features, trains and compares models with chronological walk-forward validation, stores
forecasts and later scores them against actual closes, and presents everything in a Streamlit
dashboard backed by a FastAPI service.

> **Disclaimer.** This is a research tool, **not investment advice**. Forecasts are model estimates,
> historical backtests do not guarantee future accuracy, and on daily equity data the naive
> "tomorrow = today" baseline is genuinely hard to beat. Prices are **end-of-day, delayed** data in
> Indian Rupees (₹) — nothing here is real-time.

## Contents
1. [Architecture](#architecture) · 2. [Setup (Windows/PowerShell)](#setup-windowspowershell) ·
3. [Running](#running) · 4. [Configuration](#configuration) · 5. [Methodology](#methodology) ·
6. [API](#api) · 7. [Dashboard](#dashboard) · 8. [Database](#database) · 9. [Tests](#tests) ·
10. [Example result](#example-result-real-data) · 11. [Limitations](#known-limitations) ·
12. [Troubleshooting](#troubleshooting)

## Architecture

```
frontend/ (Streamlit)  ──HTTP──▶  app/api (FastAPI + Pydantic)
                                        │
                                  app/services  (workflow, background tasks, outcomes)
        ┌───────────────┬────────────────┼──────────────────┬──────────────────┐
   app/data         app/features      app/models         app/database       app/core
 providers/ cache   engineering.py   registry/training   SQLAlchemy ORM     config, errors
 validation.py      (features,       evaluation/          + repository.py   symbols, logging
 ingestion.py        target)         prediction           (SQLite today)
 universe.py
```

* **Replaceable pieces.** Data provider: implement `DataProvider` (`app/data/providers/base.py`) and
  register it in `providers/__init__.py`. Database: change `DATABASE_URL` (SQLAlchemy; all SQL lives
  in `repository.py`). UI: the dashboard only talks to the REST API through `frontend/api_client.py`,
  so it can be swapped for React. Models: register a factory in `app/models/registry.py`.
* **Python.** Built and tested on Python 3.14 (the only interpreter on the dev machine) with pandas 3,
  scikit-learn 1.9, XGBoost 3.4. Python 3.11+ should work (`requirements.txt` uses lower bounds).

## Setup (Windows/PowerShell)

```powershell
cd path\to\price
python -m venv .venv            # or: py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1    # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements-dev.txt   # runtime deps + pytest (use requirements.txt for runtime only)
Copy-Item .env.example .env     # optional; defaults work without it
```

No API keys or credentials are needed for the default `yfinance` provider.

## Running

All commands are run from the project root with the venv active.

```powershell
python run.py init-db                          # create SQLite DB + load the small seed universe
python run.py import-universe                  # OPTIONAL: full NSE list (see note below)
python run.py fetch RELIANCE --years 8         # download & cache daily OHLCV
python run.py train RELIANCE --horizons 1 5 10 # walk-forward evaluate + save + select models (~1 min)
python run.py predict RELIANCE --horizons 1 5  # generate & store forecasts
python run.py update-outcomes RELIANCE         # record actual closes for matured forecasts

python run.py api                              # API  -> http://127.0.0.1:8000  (docs: /docs)
python run.py dashboard                        # UI   -> http://localhost:8501  (needs the API running)
python run.py all                              # API + dashboard together (Ctrl+C stops both)

python -m pytest                               # run the automated tests
```

Typical first session: `python run.py all`, open <http://localhost:8501>, pick a stock, go to
**Model Training → Start training**, then **Price Prediction → Generate forecast**. Any valid NSE
symbol can be typed into the sidebar even if it is not in the universe list; it is validated by
fetching its data.

**Search → forecast in one step.** In the dashboard, type a company name or symbol ("Infosys",
"DMART") and press Enter. If the stock has no forecast for the latest data, the app fetches prices,
trains only if there is no model younger than 30 days, and forecasts 1- and 5-day prices
automatically (about a minute the first time; instant afterwards). The API equivalent is
`POST /stocks/{symbol}/forecast` (a background task, like training). The validated forecast is
"no change" whenever no model beat the naive baseline; in that case the Overview also shows the best
trained model's price estimate in a clearly labelled **not validated** section.

**Stock universe.** The bundled list (`app/data/seed_universe.csv`, 30 large caps) is a *starter set,
not all NSE stocks*. `import-universe` downloads NSE's official `EQUITY_L.csv`
(`UNIVERSE_URL`). NSE often blocks scripted downloads; if that fails, download the file in your
browser from nseindia.com (Market Data → Securities available for trading) and run
`python run.py import-universe --source C:\path\to\EQUITY_L.csv`.

## Deploy on Streamlit Community Cloud

The dashboard can run as a single Streamlit app: if no API is reachable at the *API URL*, it starts
the same FastAPI backend inside its own process (`frontend/embedded.py`). Nothing else to host.

1. Push the repo to GitHub (the app must be readable by Streamlit; public repos are simplest).
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo, branch `main`, and set
   **Main file path** to `frontend/dashboard.py`.
3. **Advanced settings → Python version:** choose 3.12 or 3.13. Dependencies come from
   `requirements.txt` (no secrets or environment variables are required).
4. Deploy. First build takes a few minutes (XGBoost, scikit-learn, pandas).

What to expect on Cloud:
* **Ephemeral storage.** The SQLite DB, cached prices and trained models live on the container's
  disk and are lost when the app restarts or is redeployed. Every visitor shares one process and
  one database. For persistence, point `DATABASE_URL` at a hosted Postgres and store artifacts in
  durable storage (not implemented).
* **Starts empty.** No stock has models until someone trains it in the **Model Training** tab
  (about 1–3 minutes per stock on shared CPU; pick fewer algorithms to speed it up). The
  **Nifty 200 Benchmark** tab works immediately: it reads the committed `reports/` files.
* **Data source.** Prices come live from Yahoo Finance. Shared cloud IPs are sometimes rate-limited
  by Yahoo; the UI then shows the error (or stale cache) instead of fabricating data.
* **Python version.** Development and all tests ran on Python 3.14 only; the Cloud-supported 3.12/3.13
  have not been exercised here.
* The app has no authentication: anyone with the URL can trigger data fetches and training.

## Configuration

Environment variables / `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/nse.db` | SQLAlchemy URL (relative sqlite paths resolve from project root) |
| `ARTIFACTS_DIR` / `DATA_DIR` | `artifacts` / `data` | Saved models / local data |
| `DATA_PROVIDER` | `yfinance` | `yfinance` or `csv` (offline: `data/csv/<SYMBOL>.csv` with Date,Open,High,Low,Close,Adj Close,Volume) |
| `DEFAULT_HISTORY_YEARS` | `8` | History fetched when no start date is given |
| `CACHE_TTL_HOURS` | `6` | Cached data younger than this is not re-requested |
| `MIN_TRAIN_ROWS` | `250` | Minimum training rows before the first walk-forward fold |
| `WALK_FORWARD_FOLDS` / `_TEST_SIZE` | `8` / `21` | Number and length (sessions) of out-of-sample blocks |
| `LOG_LEVEL`, `API_HOST`, `API_PORT`, `API_BASE_URL` | `INFO`, `127.0.0.1`, `8000`, … | Server settings; `API_BASE_URL` is the dashboard's default API |
| `UNIVERSE_URL` | NSE `EQUITY_L.csv` | Source for `import-universe` |

**Data sources & credentials.** `yfinance` uses Yahoo Finance's unofficial public endpoints: free, no
credentials, end-of-day, unadjusted + adjusted close, subject to rate limits and occasional gaps. It is
not an official NSE feed — for production or licensed use, implement `DataProvider` for NSE or a
vendor API (credentials would then go in `.env`, which is git-ignored).

## Methodology

**Target (documented default).** For horizon *h* (default 1 = next trading day), the label at day *t*
is the relative return `y_t = P[t+h] / P[t] − 1` where *P* is the split/dividend-adjusted close. The
forecast price is `latest raw close × (1 + ŷ)`. Predicting a return instead of a price level makes
the problem roughly stationary and lets tree models work at any price level. Information used at *t*:
bars dated ≤ *t* (the last **completed** session; a bar dated today before ~16:00 IST is dropped as
possibly incomplete).

**Features** (`app/features/engineering.py`, 24, all scale-free): `ret_lag_1..5` (daily returns,
lagged closes as ratios), `mom_5/10/20`, `close_to_sma_5/10/20/50`, `sma_5_20_ratio`,
`vol_5/10/20` (rolling std of returns), `rsi_14` (Wilder), `macd_norm`, `macd_signal_norm`,
`macd_hist_norm` (12/26/9, normalised by price), `hl_range`, `gap_open` (today's open vs. yesterday's
close — known at the close of *t*), `volume_chg`, `volume_ratio_20`. **Lookback:** 50 sessions (SMA50)
— the first 50 rows have no features. Corporate-action jumps are removed by using adjusted prices.

**Leakage prevention.**
* Every feature is backward-looking (rolling/EWM windows only); a test asserts that row *t* computed
  on the full history equals row *t* computed on history truncated at *t*, and that wildly altering
  the future leaves past features unchanged.
* The target is shifted with `shift(-h)`; rows whose target is unknown are never used for training.
* Splits are chronological (never shuffled). Walk-forward folds are expanding windows; the training
  set for a fold ends `h−1` rows before the first test row so that no training target uses a price
  that is unknown at the first test prediction.
* Scalers live inside the sklearn `Pipeline` and are re-fitted on each fold's training rows only.
* Data-quality cleaning never forward-fills or invents prices: bad rows are dropped and reported.

**Models** (`app/models/registry.py`; all predict the return target through one interface):
`naive` (next close = latest close), `drift` (mean historical return), `ridge` (scaled ridge),
`random_forest`, `xgboost` (falls back to sklearn HistGradientBoosting if XGBoost is missing). No deep
learning: with ~2,000 daily rows and a near-random-walk target it would not be justified. An
ARIMA-style model was evaluated on paper and skipped — ridge on lagged returns is already an AR model.

**Metrics** (out-of-sample, per model/horizon): MAE and RMSE in ₹; MAPE % (rows with non-positive
actuals excluded; biased toward low-priced stocks); **directional accuracy** = share of rows where
`sign(pred − ref close) == sign(actual − ref close)`, counted where both moves are non-zero (naive has
no direction → shown as n/a); bias; and % improvement vs naive on MAE and RMSE.

**Model selection (predefined, never by training error).** A non-naive model is *eligible* only if,
on ≥ 40 out-of-sample rows, it beats naive by ≥ 0.5 % on **both** MAE and RMSE. The eligible model with
the lowest RMSE is selected; if none qualifies, **naive is selected** and the UI says so. Selection
among five candidates on ~168 out-of-sample rows is itself noisy — treat "beats naive" as weak
evidence, and prefer the live scoring in *Prediction History* over time.

**Multi-day forecasts are *direct*, not recursive.** Each horizon (1…20 days) has its own model trained
on that horizon's target and evaluated separately; no predicted price is ever fed back as an input
and no future actual is used. Longer horizons carry larger errors (visible in the metrics) and are
labelled as more uncertain.

**Backtest vs. forecast.** Walk-forward predictions are stored as `backtest_predictions` and shown
under "Backtest" labels; genuine forecasts are stored in `predictions` with the model version, as-of
date and reference price. `target_date_estimate` counts weekdays only (NSE holidays are not modelled),
so it is an estimate; outcomes use the actual *h-th trading bar* after the as-of date.

**Workflow (all steps implemented).** select stock → cache check → download/refresh (incremental,
with a full refetch if adjusted prices changed) → quality checks → features → train/retrain →
walk-forward evaluation → select → forecast (stored with model version) → dashboard → record the
actual close and error once available (`update-outcomes`, the dashboard button, or automatically on
the next `predict`). Everything is manual for now; a scheduler can simply call
`run.py fetch/predict/update-outcomes` daily.

## API

Interactive docs at `/docs`. Errors are always `{"error": {"code", "message", "details"}}`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Service/DB/provider status, available algorithms |
| `GET /stocks?q=&limit=&offset=` | Search the universe |
| `GET /stocks/{symbol}/history?start=&end=&refresh=` | OHLCV (+ source, last bar date, fetch time, warnings) |
| `POST /stocks/{symbol}/train` | `202` + `task_id`; body `{algorithms?, horizons?, start?, end?}` |
| `GET /tasks/{task_id}` | Task status/progress/result (in-memory; lost on API restart) |
| `GET /stocks/{symbol}/models` | Model versions, metrics, selection verdicts, feature list |
| `GET /stocks/{symbol}/models/{id}/backtest` | Walk-forward actual-vs-predicted rows |
| `POST /stocks/{symbol}/forecast` | `202`; one call: fetch data, train if needed, forecast (+ unvalidated best-model estimate) |
| `POST /stocks/{symbol}/predict` | Body `{horizons, model_version_id?, refresh_data?}`; stores forecasts |
| `GET /stocks/{symbol}/predictions` | Forecast history incl. outcomes |
| `GET /stocks/{symbol}/performance` | Realised accuracy per model vs. naive on the same rows |
| `POST /stocks/{symbol}/outcomes/update` | Fetch latest data, record actual closes/errors |
| `POST /data/refresh` | `202`; background refresh `{symbols, full}` |
| `POST /data/universe/import` | Import NSE equity list from a URL or local path |

Status codes: 404 unknown symbol/model, 409 no trained model or task already running, 422 validation,
429 provider rate limit, 502/503 provider failure or no data.

## Dashboard

Sidebar: stock search (plus free-text symbol), history range, disclaimer. Tabs: **Overview** (latest
close, data freshness/source, latest forecasts with ₹/% change, selected model metrics),
**Stock Explorer** (candlestick/line + SMA overlays, volume, OHLCV table, CSV download),
**Model Training** (algorithms, horizons, date range, live progress, comparison tables and charts),
**Price Prediction** (single/multi-horizon forecasts clearly marked as not actual prices),
**Model Performance** (backtest vs. actual, error charts, baseline comparison, metadata/features),
**Prediction History** (forecasts, actuals, errors, realised accuracy per model version).
Loading spinners, empty states and API errors are handled in the UI.

## Database

SQLite file `data/nse.db` (git-ignored). Tables: `stocks`, `price_bars` (unique symbol+date),
`data_cache_meta` (source, fetch times, quality report), `training_runs`, `model_versions` (artifact
path, features, metrics, eligibility, selection flag), `backtest_predictions`, `predictions` (unique
model_version+as_of+horizon; each references its model version), `outcomes`. Retraining never deletes
old versions or forecasts. Model files live in `artifacts/<SYMBOL>/<version>.joblib` (fitted
estimator + pipeline + feature list + training dates), also git-ignored.

## Tests

```powershell
python -m pytest
```

**Last run: 62 passed in ~40 s** (Windows, Python 3.14). The tests use a fake provider and synthetic
random-walk prices only — no network. They cover: provider responses/missing data/symbol mismatch,
cleaning and incomplete-session handling, cache and stale-fallback behaviour, feature formulas,
target shifting, no-future-information checks, chronological/purged walk-forward folds, scaler fit
scope, metric definitions (MAPE/zero handling, directional accuracy), the selection rule, model
training/persistence, prediction validity and idempotence, retraining history retention, DB
uniqueness, outcome recording, every API endpoint and error path, and a full train → predict →
outcome workflow through the API.

## Example result (real data)

`python run.py train RELIANCE --horizons 1 5 10` on 1,975 sessions (to 2026-10-09), 8 folds × 21 days
= 168 out-of-sample rows per model:

| Horizon | Selected | MAE vs naive | Comment |
|---|---|---|---|
| 1 day | **naive** | all models 0.4–2 % *worse* | No model beat "no change" — the expected honest outcome |
| 5 days | xgboost | +1.6 % | Marginal; within what 168 rows can distinguish from noise |
| 10 days | xgboost | +3.5 % | Marginal; directional accuracy 57.7 % |

These numbers are from one stock and one date; they will change on retraining. They demonstrate
the pipeline, not a profitable strategy.

## Universe benchmark (Nifty 200)

```powershell
python scripts\benchmark_universe.py --limit 200 --workers 6 --horizons 1 5   # ~2 h on 8 cores
python scripts\benchmark_universe.py --summarize-only                         # rebuild reports
```

Downloads NSE's official Nifty 200 constituent list, then per stock: (1) holds out the most recent
126 sessions; (2) trains/selects on earlier data only; (3) scores the frozen models on the held-out
window; (4) retrains on all data and stores real forecasts. Outputs in `reports/`:
`benchmark_report.md`, `benchmark_summary.json`, `forecasts.csv` (one forecast per stock/horizon),
`benchmark_results.jsonl` (raw per-stock results; resumable). The script is not covered by the unit
tests. The list contains *today's* constituents, so results carry survivorship bias.

**Result (190 evaluated of 200; 10 recent listings lacked the ≥ 457 sessions required):**

| | 1 day | 5 days |
|---|---|---|
| Naive median MAPE on held-out window | 1.33 % | 2.88 % |
| Stocks where pipeline selected naive | 173 / 190 | 149 / 190 |
| Non-naive picks that beat naive out-of-sample | 3 of 17 | 9 of 41 |
| Every algorithm vs naive (wins / losses over 190 stocks) | ≤ 60 / ≥ 130 | ≤ 41 / ≥ 149 |
| Directional accuracy of non-naive picks | 49.6 % | 49.7 % |

Conclusion: on this universe no trained model reliably beats "tomorrow = today". Models that looked
better in walk-forward validation did not hold up on the unseen window, and direction calls are a
coin flip. The pipeline's safeguard (fall back to naive) is doing its job; the forecasts are mostly
flat by design.

## Known limitations

* Data is Yahoo Finance end-of-day via an unofficial API: may be delayed, gapped, or rate-limited.
  Adjusted prices are revised by the provider after corporate actions (handled by full refetch when a
  change is detected in the overlap window).
* NSE trading holidays are not modelled; `target_date_estimate` is weekday-based. Outcomes are exact.
* Task status is in-memory (not durable across API restarts); training rows persist in the DB.
* Features are price/volume only (no fundamentals, news, index or macro data); expect modest skill.
* The notebook from the suggested layout was intentionally not created (it would have been an empty
  shell); exploration is covered by the dashboard.
* Intraday data, scheduling, authentication and PostgreSQL migrations (Alembic) are not included.

## Troubleshooting

* **Dashboard says it cannot reach the API** → start `python run.py api` (or `python run.py all`) and
  check the *API URL* in the sidebar.
* **`Activate.ps1` is blocked** → `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
* **Rate-limited (HTTP 429) / "No data" from Yahoo** → wait a few minutes; cached data keeps working
  (the UI flags it as stale). Invalid or delisted symbols return a clear 404.
* **`import-universe` fails** → NSE blocks scripts; download `EQUITY_L.csv` manually and pass
  `--source <path>`.
* **"Not enough history"** → the stock needs roughly `MIN_TRAIN_ROWS + 21 + 50` clean sessions;
  increase the history length or lower `MIN_TRAIN_ROWS`/`WALK_FORWARD_FOLDS` for recent listings.
* **Training is slow** → train fewer algorithms/horizons; XGBoost and random forest dominate runtime.
* **Port already in use** → `python run.py api --port 8001` and `python run.py dashboard --port 8502`;
  set `API_BASE_URL` (or the sidebar field) to match.
* **Reset everything** → delete `data\nse.db` and `artifacts\*` and run `python run.py init-db`.
