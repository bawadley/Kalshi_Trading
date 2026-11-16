# V7.3 — short-horizon order-flow forecasting (research only)

V7.3 evaluates **whether order-book and trade-flow features improve BTC 15-minute contract-price predictions**, instead of assuming that more messages automatically mean a profitable strategy.

## Run from PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_v73 -v
.\.venv\Scripts\python.exe v73_orderflow_models.py --horizons 5,15 --train-markets 6 --val-markets 2 --test-markets 2
```

Requires the previous `data/v72/btc_1s_features.csv`. Regenerate it after collecting more markets if desired:

```powershell
.\.venv\Scripts\python.exe v72_features.py
```

## What it compares

- **No-change baseline:** forecasts a $0 change in the YES bid/ask midpoint.
- **Price-only:** `mid` and `spread`.
- **Price + order flow:** price features plus 5-level YES/NO depth, order-book imbalance, microprice, top-of-book quantities, trade frequency/volume, taker outcome volumes, delta frequency, and quote age.
- Each feature set tries Ridge, Extra Trees, and histogram gradient boosting. The **validation** market block picks a model. The following **test** block is evaluated only afterward.

The default expanding chronological splits are six markets for initial training, two subsequent markets for validation, two later markets for testing, then repeat by advancing the test block by two markets. With 13 markets, only around two complete test folds are expected (the last partial fold is not included). Model results from two folds are **exploratory**.

Within each test market, the module evaluates at timestamps spaced **at least the prediction horizon apart**, reducing overlapping target windows in reported test metrics. Training data still contains overlapping one-second labels. Market boundaries stay intact and aren't randomized.

Default `--min-rows-per-market 20` excludes markets with too few horizon-specific labels; this may reduce eligible folds especially at longer horizons.

## Reports

- `reports/v73/v73_fold_metrics.csv` — per-horizon and per-fold MAE, validation-selected model, feature set, no-change MAE and improvement percentage.
- `reports/v73/v73_predictions.csv` — prediction vs actual movement for each non-overlapping test timestamp.

Files in `reports/` are gitignored locally; historical raw data and prior V7.2 artifacts are unchanged.

## Essential caveats

- The one-second features are computed from all events observed *during* a given second, so a prediction at that timestamp is only feasible **after that second closes**, plus observation and processing latency. The evaluation doesn't yet explicitly simulate that extra latency.
- The data was collected with **local receipt timestamps**, not perfectly synchronized exchange timestamps. Some reconnections and periods with no messages may be missing.
- The V7.2 extractor intentionally doesn't carry order-book state across files or sequence gaps; targets are constrained within those boundaries.
- The same BTC markets may already have influenced earlier model development, so reported performance on them is *not* an independent investment-grade out-of-sample result.
- Only ~13 distinct markets were captured initially. Thousands of one-second records are highly correlated, and choosing model variants may overfit validation markets.
- **No trade execution or profit claims.** Better midpoint MAE does not establish tradable bid/ask edge, queue priority, fill probabilities, or real returns.

Keep the V7.2 recorder running separately and collect additional distinct BTC market windows for future clean evaluation.
