# Kalshi Bitcoin V7.2 — continuous market recording + event features

V7.2 fixes V7.1's scheduled WebSocket disconnects. It discovers the active BTC 15-minute market **concurrently** with a healthy connection. Only market rotation, connection failure, or an order-book sequence gap ends the session. Each session gets its own raw recording file and starts with a fresh full snapshot.

## Update and test (PowerShell, in the project directory)

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_v72 -v
```

No new packages beyond V7.1 are required (`requests`, `pandas`, `websockets`, `cryptography`).

## Use V7.2 instead of the V7.1 collector

Stop the old `v71_rolling_ws.py` using Ctrl+C, then start V7.2:

```powershell
.\.venv\Scripts\python.exe v72_rolling_ws.py --series KXBTC15M --seconds 0 --discovery 15
```

- `--seconds 0` runs while the process and PC remain active.
- `--discovery 15` checks new markets every 15 seconds **without** reconnecting a healthy book socket.
- Raw files go into `data/v7/`. No historical files are deleted, truncated, or overwritten. There is no background service or cloud scheduling.
- A single market session file may become large. Monitor available storage and OneDrive syncing.

## Convert your existing 332 JSONL files into a compact dataset

Stop the collector, or run the processing command in another terminal (active files can change while being read; completed files are preferable):

```powershell
.\.venv\Scripts\python.exe v72_features.py
```

This produces:
- `data/v72/btc_1s_features.csv`: one-second last-known bid/ask, spread, 5-level depth per side, imbalance, microprice, event count, trade count and volume, taker-outcome volume, and 5/15/30/60 second **future MIDPOINT CHANGE** labels.
- `data/v72/btc_integrity.csv`: audit for each raw file (snapshots, applied/invalid deltas, bad JSON/timestamps, reconstructed seconds and count of labeled rows).

The extractor streams each file's events and holds only ~one session's one-second rows in memory. All raw data remains untouched. Output CSVs are recreated whenever processing is rerun.

### Dataset integrity policy

- A file starts unsynchronized and must receive a full order-book snapshot before quotes are accepted.
- Sequence gaps or negative order-book sizes invalidate reconstructed depth until the next snapshot. Invalid seconds are excluded.
- Forward-price targets never cross recording files, and require a contiguous second-by-second quote sequence.
- Price features represent a second's closing observed book, **not** a quote that would be known at the start of that second. Trade counts reflect events observed that second.
- Times use local receipt clocks, so network delays, exchange clocks and synchronization error remain to be measured.
- `taker_outcome_side` is categorized as YES/NO/unknown where provided; it does not identify maker orders or guarantee actual buying pressure.
- Historical V7.1 capture gaps cannot be recovered retroactively. A file with no sequence error can still miss data before its initial snapshot or after the end.
- Repeated 1-second samples within a 15-minute market are highly correlated. Split ML datasets by complete market **and chronological time**, not by random sample rows. Reserve future markets as genuinely unseen tests.
- This module does not claim any profitable strategy or estimate real fills.

## What to check

After running the extractor, inspect `data/v72/btc_integrity.csv` for invalid deltas and missing seconds. A zero invalid-delta count is helpful but **not** proof of completeness. Subsequent model experiments should compare against a no-change baseline and include execution-cost stress tests.
