# Bitcoin V6 — walk-forward, selective trading research

This version builds on V5's timestamp-safe BTC spot/index-proxy features and contract strike metadata. **It is read-only and never places real orders.**

## Setup (PyCharm PowerShell)

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_btc_v6 -v
.\.venv\Scripts\python.exe btc_v6_strategy.py --series KXBTC15M --hold 2 --min-train-markets 250 --validation-markets 100 --test-markets 50 --slippage 0.01
```

Requires existing V5 data:
- `data/KXBTC15M_minute.csv`
- `data/v5/KXBTC15M_metadata.csv`
- `data/v5/coinbase_btcusd_1m.csv`

If missing, first run `kalshi_lab.py fetch --series KXBTC15M --limit 1000` and `btc_v5_data.py --series KXBTC15M --days 14`.

## Method

1. Signal at end of minute t; hypothetical entry at the ask of t+1; exit at the bid at t+1+`hold`.
2. Future prices and realized hypothetical returns are training labels only, **never forecast features**.
3. Price forecasting: histogram-gradient-boosting regressor on the V5 BTC spot, strike (only if verified), countdown, and market features.
4. Profitability estimates: separate random-forest classifiers for YES and NO trained on historical profitable/unprofitable hypothetical round-trip labels, including fees and slippage.
5. Every expanding walk-forward fold uses earlier markets for training, immediately later markets for validation/threshold selection, and subsequent markets for score-only test. A new test fold advances by `--test-markets`.
6. On the validation window only, search conservative predicted-edge and probability thresholds, requiring at least `--min-validation-trades`. No positive validation candidate means **NO TRADE** for that fold.
7. On test, select the earliest qualifying YES or NO opportunity in each market. Maximum one hypothetical trade per market.
8. Reports: `reports/KXBTC15M_v6_folds.csv` and `reports/KXBTC15M_v6_trades.csv`. Empty trade reports are valid.

## Interpreting results carefully

- Reported dollars are **quote-proxy P&L**, not actual executable results. We do not know book depth, position in queue, latency, missed quotes, trade prints, or real fills.
- Classifier probabilities are **not calibrated**; 0.65 output is not guaranteed to mean 65% real chance.
- Fee calculator uses an illustrative taker formula with per-order rounding; verify the actual BTC fee table.
- Coinbase spot isn't the CF Benchmarks index Kalshi uses for settlement.
- The validation-threshold search may still overfit the validation interval, even when the forward tests are chronologically separated.
- Earlier V1–V5 research analyzed overlapping historic windows, so the existing 1,000-market sample is **not a pristine unseen benchmark**. Collect later markets before evaluating investable performance.
- This intentionally doesn't auto-place trades, train a sequence-based deep net, or claim an event-based order-book simulator. Those need real authenticated market event recordings.

Further milestones: calibrated probabilities, walk-forward stability/CI analysis, sensitivity to latency/slippage, reliable order-book and trade-event feeds, a paper-trading simulator with real order availability and inventory constraints.
