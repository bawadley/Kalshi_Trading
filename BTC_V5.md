# Bitcoin V5 — better data and forward testing

V5 is a read-only research upgrade alongside the existing V1–V4 files. No live orders.

## Setup in PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe kalshi_lab.py fetch --series KXBTC15M --limit 1000
.\.venv\Scripts\python.exe btc_v5_data.py --series KXBTC15M --days 14
.\.venv\Scripts\python.exe btc_v5_walkforward.py --series KXBTC15M --horizon 2 --min-train-markets 150 --fold-markets 50
```

Data collector saves:
- `data/v5/KXBTC15M_metadata.csv` with Kalshi per-market status, open/close timestamps, settlement result, verified strike fields when present, rule text and contract title;
- `data/v5/coinbase_btcusd_1m.csv` with Coinbase Exchange BTC-USD open/high/low/close/volume and an `available_ts` marking **minute end**, not candle start.

The evaluation script makes time-safe joins with `direction=backward` and a maximum spot age. It derives spot momentum, range, volume, Kalshi midpoint, spread, recent contract-price changes, minutes to close, and **strike distance only if both training and test rows have an explicitly numeric market strike**. It does not infer strikes from titles, prices, expired outcomes, or other future data.

The output `reports/KXBTC15M_v5_walkforward.csv` contains fold-wise test prediction errors for an expanding-window gradient-boosted regressor and a no-change baseline. No fills or profit statistics are asserted.

## Important limitations

- **Coinbase BTC-USD is not Kalshi's actual CF Benchmarks settlement index.** It may differ from the exact reference calculation and cannot establish contract outcomes by itself.
- Some Kalshi contracts do **not** supply a numeric floor strike through the public market endpoint. Missing values remain missing. No strike-based prediction is made until the exact market specification is verified.
- Public minute candlesticks have gaps and no order-book queue/fill data. Spot candles are only usable at or after their completed minute; research performance may still be optimistic because of exchange latency.
- Expanding walk-forward folds are better than a single random split, but markets previously inspected in V1–V4 aren't genuinely new. Reserve later, newly collected market windows as a final unseen evaluation set.
- This V5 script is **forecast-only**; the existing `btc_selective_v4.py` remains the separate, explicitly approximate execution research tool.
- Collecting richer event data next requires Kalshi's authenticated orderbook_delta WebSocket (snapshot + sequence-controlled deltas), trade feed, and reference-price history. Do not claim genuine market-making fills from REST book snapshots.

## Diagnostics

If you get `Need N usable market windows`, fetch more data and/or lower `--min-train-markets` and `--fold-markets` for a software check (not for final research claims).

If few rows survive, verify Coinbase spot file timestamps overlap the Kalshi minute data date range. To verify metadata independently:

```powershell
.\.venv\Scripts\python.exe btc_v5_data.py --series KXBTC15M --days 14 --metadata-only
```

Source references:
- https://docs.kalshi.com/
- https://docs.kalshi.com/websockets/orderbook-updates
- https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
