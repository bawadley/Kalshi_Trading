# V7.4: rolling order-flow features + delayed Bitcoin spot data

This is a **read-only exploratory price-forecasting** upgrade. V7.2's continuous
recorder stays unchanged, and V7.3 remains available for comparison.

## Run in your PyCharm PowerShell terminal

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_v74 -v
.\.venv\Scripts\python.exe v74_orderflow.py --horizons 5,15 --train-markets 6 --val-markets 2 --test-markets 2
```

Input datasets:
- `data/v72/btc_1s_features.csv` from `v72_features.py`
- `data/v5/coinbase_btcusd_1m.csv` from `btc_v5_data.py`
- `data/v5/KXBTC15M_metadata.csv` from `btc_v5_data.py`

If you have captured more markets, regenerate your V7.2 feature CSV first using
`python v72_features.py` when the recorder is stopped or by processing only
completed recording files.

## What is tested

Four *matched* feature sets on exactly the same market windows:
1. `price_only`: Kalshi midpoint + spread.
2. `price_flow`: price plus current depth, imbalance, microprice and trade flow.
3. `price_flow_dynamics`: adds lagged 5/15-second changes in midpoint, depth and imbalance, rolling trade counts/YES-minus-NO taker volumes, and midpoint volatility.
4. `price_flow_dynamics_btc`: adds conservative Coinbase spot returns, its distance from the exact supplied Kalshi numeric floor strike, seconds to close, and spot age.

The Coinbase bar's `available_ts` means end of the one-minute bar; a **further 60-second safety delay** is applied by default, controlled by `--spot-lag`. The join is backwards in time only. These prices are a **proxy, not Kalshi's settlement CF Benchmarks price**. The recorded contract floor strike is treated as a fixed contract term; verifying exactly when the strike and any revisions were first published is additional required research before live trading.

Rolling features are computed *within* each recording file, market, snapshot epoch, and contiguous one-second segment. No forward-looking values are used as predictors. Missing rolling history remains missing; it is not filled using future observations. Models impute missing predictors using training-data statistics only.

For 5- and 15-second target horizons, use expanding training markets, the immediately following validation markets for **model selection**, and subsequent held-out test markets. Predictions are evaluated at non-overlapping signal intervals to reduce dependence.

The three model candidates per feature set are Ridge, Extra Trees, and gradient-boosted trees. All selected models report mean absolute errors against a no-change baseline.

## Outputs

- `reports/v74/v74_enriched_features.csv` — all one-second rows and added features.
- `reports/v74/v74_fold_metrics.csv` — model comparisons for each fold and horizon.
- `reports/v74/v74_predictions.csv` — individual held-out forecasts, true midpoint changes and a **spread + 2-cent directional threshold screen**.

The threshold screen reports how often predicted movement exceeds the existing spread plus a buffer. It **does not** measure order fills, actual fees, executed profits, maker queue position or position risk. All four feature sets must be evaluated on common test markets, and results remain exploratory with ~13 markets.

## Cautions

- Price forecasts are measured using a second's final observed book; receiving and processing all events costs real time.
- Train windows still have overlapping labels; test signals are less overlapping, but individual seconds within a 15-minute market are highly dependent.
- A sample of 13 Bitcoin markets is inadequate for declaring a reliable strategy.
- Models may perform worse than predicting no midpoint change; that is a valid outcome.
- Do not treat the spread-plus-2-cent diagnostic as a genuine trading strategy. The eventual execution simulator needs actual exchange message timing, available liquidity, order fees and latency modeling.

Next milestone: more independent markets, audit latency and order-book completeness, evaluate one-sided executable-price targets and calibrated probabilities before any paper-trading.
