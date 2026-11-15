# Bitcoin 15-minute dual predictor (read-only)

`btc_predictor.py` treats **two separate questions**:

1. **Settlement**: P(YES ultimately settles to $1) from observed contract quotes/features.
2. **Intrawindow**: forecast the market YES midpoint 1, 2, or 3 minutes later. An optional trade proxy tests hypothetical long-YES buying at the *following minute's ask* and liquidating at the *future bid*.

## From PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe kalshi_lab.py fetch --series KXBTC15M --limit 500
.\.venv\Scripts\python.exe btc_predictor.py --series KXBTC15M --mode all --horizons 1,2,3
```

You can isolate one task using `--mode outcome` or `--mode price`.

Output files (ignored by Git):
- `reports/KXBTC15M_outcome_predictions.csv` contains predicted settlement probabilities and historical outcomes.
- `reports/KXBTC15M_intrawindow_predictions.csv` contains predicted future contract midpoint, actual future midpoint, long-YES signal, and **hypothetical** quote-based net per contract.

Model comparisons:
- Settlement: logistic regression, random forest, histogram gradient boosting, MLP neural network; report Brier score and AUC, compare with Kalshi midpoint baseline.
- Price: Ridge, Extra Trees, MLP neural network; choose via **validation** MAE; compare to predicting no change.
- Train/validation/test split by chronological market window (60/20/20); do not randomly split individual 1-minute rows.

**Key limitations and next work:**
- This historical starter uses **minute quote data only**; no verified order execution, queue rank, slippage, funding, actual fill times, and no direct BTC spot features. A quoted 30¢→50¢ midpoint move does not guarantee buying at 30¢ and selling at 50¢.
- Historical YES/NO labels are taken from settled market result strings; unlabelled windows are skipped by settlement model. Prices at test time are used only for score computation, never as features.
- Fees use an **illustrative** formula; verify actual applicable series fees. It always excludes uncertain trades from actual performance claims.
- The minute-quote research may use overlapping signals; trade proxy P&L is not risk-constrained or capacity-adjusted. Also, consecutive 15-minute windows could overlap in time in practice; ensure the chosen set has no purging/embargo problems before treating test statistics as definitive.
- An outcome calibration study should use one observation per specified time-to-expiry band per window (to avoid overweighting windows with more candles).
- Next stage: timestamped BTC spot/index prices, settlement reference feed, market open/close countdown, event-time book deltas, trade prints, queue-aware execution, NO-side strategies, actual fees, and probability calibration. Only then consider paper trading. No account credentials or live orders.
