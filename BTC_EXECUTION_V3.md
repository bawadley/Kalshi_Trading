# BTC execution research V3 — corrected timing and two-sided trading

This **read-only** quote-proxy backtester is intentionally separate from the earlier predictor. It fixes the previous same-timestamp one-minute entry/exit mistake and avoids overlapping simulated positions within a market.

## Run in PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe btc_execution_v3.py --series KXBTC15M --hold 2 --threshold 0.06 --slippage 0.01 --contracts 1
```

For a one-minute holding period, set `--hold 1`. The simulation makes a forecast at the close of minute t, assumes entry at the **following** minute's quote, then exits **hold full minutes after entry**. The model only sees fields available through t, never its future entry/exit prices. A market may produce **at most one** simulated trade. Positive predictions can buy YES; negative predictions can buy NO. The entry is a quoted ASK and the exit is a quoted BID for the selected side.

Models: Ridge, Extra Trees, histogram gradient-boosted trees. Model selection uses a chronological 60/20/20 training/validation/test split. Only the validation set determines the selected model (lowest validation MAE). The threshold is supplied in advance, not optimized on the test set. The held-out test is used only for a final diagnostic.

Reports: `reports/KXBTC15M_v3_trades.csv` includes hypothetical entry/exit timestamps, side, quoted prices after an adverse-slippage adjustment, estimated fees, and simulated P&L.

**Be careful:** the entry at the next minute is an optimistic observation-based execution assumption, not proof that an order could fill. Likewise, the later bid may not have had enough volume. Real trading also has latency, fees that depend on series and quantity, and occasional markets where a quote moves before we act. The default taker fee estimate uses the July 2026 standard fee formula with centicent rounding, but verify that the actual BTC series fee multiplier matches. This module has **no** true maker fill simulator or order-book event replay. Do not treat any P&L from it as executable profits.

Kalshi fee schedule: https://kalshi.com/fee-schedule

Roadmap: integrate BTC spot/index target and clock-to-settlement, event-level L2 book and prints, conservative fill simulation with latency and slippage distributions, rolling walk-forward evaluation, and calibrated expected returns. Do not run live orders before paper-trading these.
