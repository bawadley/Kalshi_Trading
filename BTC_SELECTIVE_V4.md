# Bitcoin selective V4: YES / NO / NO TRADE

**Research-only.** Adds `btc_selective_v4.py` without changing the previous working modules.

## In PyCharm PowerShell
```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe btc_selective_v4.py --series KXBTC15M --hold 2 --slippage 0.01 --contracts 1
```

The algorithm:
1. Generates historical signal features and entry/exit quote targets with the prior V3 timeline: signal at t, hypothetical entry at t+1 minute, exit after `hold` full minutes.
2. Chronologically splits by market 60% training, 20% validation, 20% held-out test.
3. Fits two price regression candidates and selects one based ONLY on validation forecast MAE.
4. Trains separate Random Forest classifiers to estimate whether a YES or NO quote-based round trip has positive simulated net P&L. Positive trade labels use **training-only future quotes**. If the class balance is inadequate, that side is disabled.
5. Searches validation-only candidate predicted-edge and profitability-probability thresholds, requiring a minimum trade count. If no threshold has positive aggregate validation returns, it **does not trade** in the held-out test.
6. Takes the first qualifying trade per market, at most one position in each market, and reports hypothetical quote-based proceeds net of an assumed taker fee and per-side adverse slippage.

The model's `probability` is a raw ML classifier output; it is **not calibrated** and must not be interpreted as a reliable real-world probability without a calibration study. The price edge is an estimated quote-based ranking indicator, not expected execution P&L. Selecting a threshold using validation P&L may overfit that set. The test set remains untouched until selection, but previous V1-V3 model development has already referenced the same historical market periods; use **newer never-before-used markets** for independent validation.

If permitted by the validation gate, exports `reports/KXBTC15M_v4_selective_trades.csv`. Otherwise exports an empty report with a clear no-trade message.

**Limitations**: This is NOT a realistic exchange simulator: quote candles don't prove order executions or depth; fee schedule must be checked for the BTC series; model class probabilities are uncalibrated; the simulator does not model cancellations, order delays, price impact, settlement/forced exits, or actual live P&L. Never connect it to real-money order placement without an independently validated execution model and a paper-trading stage.

Further work: join timestamped BTC reference price and strike, spread and L2 event streams, candidate trade ranking using calibrated expected P&L, walk-forward folds with purging/embargo, and fee-sensitive stress tests.
