# V7.1 — Automated read-only Kalshi BTC WebSocket recording

V7.1 rotates across active KXBTC15M markets without requiring a new ticker every 15 minutes. It **does not place orders** and does not run the V6 predictor.

## Install and test (PyCharm PowerShell)

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m pip install "websockets>=14,<17" "cryptography>=42" requests
.\.venv\Scripts\python.exe -m unittest test_v71_rolling -v
```

Use the same `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` environment variables configured for V7. Never paste private keys into chat or commit them.

## Record for one hour

```powershell
.\.venv\Scripts\python.exe v71_rolling_ws.py --series KXBTC15M --seconds 3600 --discovery 30
```

For unlimited recording **while that terminal and computer are running**:

```powershell
.\.venv\Scripts\python.exe v71_rolling_ws.py --series KXBTC15M --seconds 0 --discovery 30
```

Press **Ctrl+C** to stop. Output is in `data/v7/`, with one file per bounded session. The recorder caps messages per file (`--max-messages 150000` by default) and reconnects to resume. It may create several files for the same 15-minute market: this is intentional. Each session starts with a fresh subscription and snapshot. During market transitions there may be unavoidable recording gaps.

**Disk and data rates:** Some markets send tens of thousands of messages per minute. Storage use may be large. Monitor available disk space before extended collection. The script does not run in the cloud or when your computer is sleeping.

## Check what has been captured

```powershell
.\.venv\Scripts\python.exe v71_stats.py --latest 5
```

For all saved files (can be slow if many gigabytes are stored):

```powershell
.\.venv\Scripts\python.exe v71_stats.py
```

Replay a single file as with V7:

```powershell
$file = (Get-ChildItem .\data\v7\*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe v7_replay.py --input "$file" --contracts 1 --hold-seconds 60
```

## Integrity and limitations

- Uses Kalshi REST `/markets` to discover an open individual ticker, choosing the earliest future close time.
- Opens authenticated WebSockets with the existing Ed25519/RSA signer from V7.
- Stores full snapshots, subsequent deltas and trade prints with **local receipt timestamps** in JSONL.
- On unexpected sequence discontinuity, invalidates the captured book and reconnects; the offline replay must start from the new snapshot. Messages during reconnection are lost.
- Market discovery uses a polling interval, not exchange push notification; transitions can have gaps.
- The collector subscribes only to public market data and never submits a trade. The recorder does not itself calculate or verify profits, market making fills, fees, or queue positions.
- The stats tool counts events; separate replay validates book reconstruction. Do not interpret zero counted errors as proof of data completeness.
- Recording files may contain trading data and account-adjacent metadata. Keep file permissions appropriate; the recorder does not write your PEM key or its contents to the output.

Current scope: automated rotation and capture statistics. Planned next: regularized event-flow features, robust reconnect watermarking, rotation gap auditing, and V6 signal alignment under realistic latency.
