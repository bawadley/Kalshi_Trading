# Kalshi Trading V7 — event-level BTC market data and replay

This version adds a **read-only authenticated WebSocket recorder** and an **offline depth replay**. Existing V1–V6 models are unchanged. No order placement.

## Setup in PyCharm (PowerShell)

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install "websockets>=14,<17" "cryptography>=42"
.\.venv\Scripts\python.exe -m unittest test_v7_replay -v
```

Create a **read-only / market-data API key** through Kalshi's account API settings. Keep its private PEM key outside this repository. Never paste it into chat, commit it to GitHub, or send it to anyone. Set credentials in the **current terminal session**:

```powershell
$env:KALSHI_API_KEY_ID="YOUR_READ_ONLY_KEY_ID"
$env:KALSHI_PRIVATE_KEY_PATH="C:\path\outside\repo\kalshi_readonly.pem"
```

Find a currently active *individual* BTC market ticker:

```powershell
.\.venv\Scripts\python.exe find_markets.py --series KXBTC15M
```

If that helper returns none, query Kalshi's public markets endpoint and find markets with status `active`. Do not pass the series ticker itself to the recorder.

Record the chosen market for up to 15 minutes:

```powershell
.\.venv\Scripts\python.exe v7_ws_recorder.py --ticker YOUR_ACTIVE_BTC_MARKET_TICKER --seconds 900
```

The file appears under `data/v7/YOUR_ACTIVE_BTC_MARKET_TICKER_<epoch>.jsonl`. The recorder writes raw authenticated `orderbook_snapshot`, `orderbook_delta`, and `trade` messages with local receipt timestamps. It uses market-data subscriptions only, and reconnects with a fresh snapshot after connection loss. A delta sequence gap marks the local book unreliable and requests `get_snapshot`.

Replay a saved file:

```powershell
.\.venv\Scripts\python.exe v7_replay.py --input data/v7/YOUR_FILE.jsonl --contracts 1 --hold-seconds 120 --slippage 0.01
```

**Meaning of replay**: it reconstructs aggregate YES/NO bid levels in sequence and checks whether displayed depth supports a *hypothetical taker entry/exit*. It runs an illustrative first-available YES trade per market, not a learned strategy. No queue rank, maker fill, actual submitted order, trade latency, exchange matching confirmation, or guaranteed liquidation. The fee estimate is illustrative and doesn't round per actual exchange order. The output is a **quote-depth stress diagnostic**, not investment-grade P&L.

### Current limits and next work

- The recorder captures one manually chosen market; it does not yet rotate automatically across 15-minute BTC windows.
- The replay uses local receive timestamps; exchange timestamps must be incorporated for rigorous latency-aware simulation.
- Data capture cannot reconstruct earlier events before subscribing. Order flow is partially observable and individual trader identities are not exposed.
- Kalshi book feeds use YES/NO bid levels and `seq` on a subscription to detect missing messages; on a gap use a **new snapshot**, not silent continuation.
- Connection loss invalidates preexisting state. An empty snapshot can indicate a determined market.
- Future V7.1: automated market rotation, event timestamp normalization, stronger disconnect/resync markers, and paper-order accounting.
- Future V8: combine live BTC spot/reference data and trained predictive signals with this event replay, including partial-fill assumptions.

Documentation:
- https://docs.kalshi.com/getting_started/quick_start_websockets
- https://docs.kalshi.com/websockets/orderbook-updates
