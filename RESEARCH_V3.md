# Research V3: market depth and microstructure

**Read-only development stage. No live trading or verified historical maker fills.**

The previous `kalshi_lab.py` / `research_v2.py` remain unchanged.

## Run from PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### A. Record real snapshots

Find an **open individual market ticker**, not the series ID `KX2YRRATE15M`. A series contains multiple 15-minute markets. For example, find an open market ticker through Kalshi's website/API and substitute its exact identifier:

```powershell
.\.venv\Scripts\python.exe market_depth.py collect --ticker REPLACE_WITH_OPEN_MARKET_TICKER --seconds 300 --interval 2
```

This records publicly displayed order-book depth to `data/depth/<market_ticker>.jsonl`. It does not record past markets retroactively. Append one file per ticker; do not concatenate disconnected time periods without tracking gaps.

### B. Engineer depth features

```powershell
.\.venv\Scripts\python.exe market_depth.py features --input data/depth/REPLACE_WITH_OPEN_MARKET_TICKER.jsonl
```

Derived variables include: top-of-book spread and sizes, microprice, 1/3/5/10/20-level buy/sell liquidity, depth imbalance, level-weighted prices, largest walls, 2,500-contract wall flags, bid/ask concentration, momentum, volatility, spread changes, wall persistence and quote update intervals. Yes ask is derived as `1 - best NO bid`. This is **aggregate displayed contracts per price level**, not individual order counts or trader identities.

### C. Evaluate models after collecting many markets

```powershell
.\.venv\Scripts\python.exe depth_models.py --input reports/depth_features.csv --horizons 30,60,120,180,300
```

The model requires features containing **30 or more distinct market tickers** and time horizons with sufficiently dense snapshots. It will intentionally refuse to fit on a single market. You can merge the per-market CSV files into one `reports/depth_features.csv` when many markets have been recorded (preserve the original `ticker` and timestamps).

Compares Ridge, Random Forest, Extra Trees, and MLP neural network. It predicts both future midpoint change and magnitude of price movement. Train/validation/test are ordered by market start (60/20/20), never randomly shuffled. Outputs `reports/depth_model_predictions.csv`. Model rankings must be based on validation, followed by a single untouched test. No fee-adjusted P&L is claimed from these forecasts.

### D. Market-making quote-crossing stress test

```powershell
.\.venv\Scripts\python.exe market_depth.py stress --input data/depth/REPLACE_WITH_OPEN_MARKET_TICKER.jsonl --hold 120
```

A **diagnostic upper-bound proxy only**, not event replay. It requires subsequent opposite-side quotes to cross our hypothetical levels, but has **no actual trade prints, queue position, order priority, cancellation latency, partial fills or maker fee validation**. Its gross proxy profit must not be treated as backtest profit or an investable return. It is possible for displayed quote crossings not to fill our orders.

## Data that we cannot infer from snapshots

- Individual order count, owner identity, iceberg orders and queue rank
- Every between-sample cancellation / amendment / fill
- Real order arrival latency or cancel acknowledgement
- Bid/ask volume actually executed against our position
- Underlying Treasury yield/reference-index update timestamps
- Exchange market rules, settlement source, fees and quote incentives unless sourced separately

For a **true event-driven simulator**, record authenticated Kalshi WebSocket `orderbook_snapshot` and `orderbook_delta` messages, public trade prints and exchange timestamps. Validate sequence gaps and resync on reconnect; use event-time replay, conservative FIFO queue assumptions, maker/taker fees, inventory, risk limits, and settlement. Do not count maker fills without sufficient execution evidence.

Docs:
- https://docs.kalshi.com/api-reference/market/get-market-orderbook
- https://docs.kalshi.com/websockets/orderbook-updates

Never commit secrets or bulky downloaded data. `.gitignore` excludes `data/`, `reports/`, `.env`.
