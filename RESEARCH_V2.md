# Research V2: 15-minute forecasting and market-making proxy

This is a **read-only educational research tool**, not a trading bot. It never sends orders and requires no trading credentials.

## Quick start (PyCharm PowerShell)

From your cloned repository:

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research_v2.py --series KX2YRRATE15M --mode all
```

Use the historical minute data previously downloaded by `kalshi_lab.py fetch`. The new module DOES NOT download market data itself.

```powershell
.\.venv\Scripts\python.exe research_v2.py --mode forecast --horizons 1,2,3,4,5
.\.venv\Scripts\python.exe research_v2.py --mode volatility
.\.venv\Scripts\python.exe research_v2.py --mode profitability
.\.venv\Scripts\python.exe research_v2.py --mode maker
```

### Modules
- **Forecast**: Multi-output one- to five-minute YES midpoint predictions using Ridge, random forest, extra trees, MLP neural network, and a validation-selected two-model average. Saves `reports/candle_forecasts.csv`, with a separate actual future midpoint for each requested horizon.
- **Volatility**: Predicts the sum of absolute minute-to-minute midpoint movements over each future horizon; compares a historical-median baseline against random forest and MLP.
- **Profitability**: Predicts whether a hypothetical purchase at the following minute's YES ask and sale at the later bid exceeds an illustrative taker fee. Compares logistic regression, random forest, neural classifier, and a validation-selected ensemble. Reports AUC and hypothetical net return at a fixed 65% predicted-probability threshold.
- **Maker**: Tests hypothetical one-tick-inside bid/ask quotes. An assumed maker buy requires a *later displayed ASK* to cross below the placed bid by at least one tick. A later displayed BID must cross above the maker ask by one tick for a simulated matched exit. Otherwise the proxy attempts a bid-side forced liquidation. This is stricter than mere quote touch, but STILL DOES NOT demonstrate actual fills.

### Design boundaries
- All feature columns are known at the end of the signal minute. Future asks/bids are **labels/score-only**, never training features.
- A market-level chronological 60/20/20 split isolates train, model selection, and test. Adjacent minute samples within each market must have no gaps across the full forecasting window.
- Sample predictions in overlapping windows and repeated trades are not independent; profitability P&L **does not constrain capital, enforce position exclusivity, or account for execution/latency**. Therefore the P&L columns **must not** be interpreted as realistic trading results.
- Historical candles do not contain order-book depth, queue position, or actual maker fill timestamps. The maker model is a **quote-crossing stress proxy**, not an event-driven replay.
- Fee function uses an illustrative taker fee formula, not a verified current contract-specific schedule. Maker fee defaults to zero but should be changed if needed.
- No live account access, no automated trading, no API keys.

### Next steps for a proper backtest
1. Collect a historical stream of L2 book deltas and executed trades for `KX2YRRATE15M`.
2. Reconstruct queue positions and enforce a matching-engine fill policy, including cancellation and latency.
3. Measure calibration and profitability on rolling *out-of-sample* time windows.
4. Add underlying Treasury yield movement and public-data timestamps, ensuring no lookahead.
