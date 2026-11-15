"""V7.1 read-only rolling Bitcoin event recorder.

Rotates automatically across open KXBTC15M markets using public REST discovery.
Every WebSocket session writes to its own JSONL file under data/v7.
Never places orders; supports Ed25519 and RSA via v7_ws_recorder.headers.
"""
import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import websockets

from v7_ws_recorder import WS, OUT, headers

REST = "https://external-api.kalshi.com/trade-api/v2"


def close_epoch(market):
    raw = market.get("close_time") or market.get("expiration_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def choose_market(markets, now=None):
    """Earliest closing currently tradable market; never choose an expired one."""
    now = time.time() if now is None else now
    eligible = []
    for m in markets:
        t = m.get("ticker")
        close = close_epoch(m)
        if t and close is not None and close > now and m.get("status") in ("active", "open"):
            eligible.append((close, t, m))
    return min(eligible, default=(None, None, None), key=lambda x: (x[0], x[1]))[2]


def discover(series):
    r = requests.get(REST + "/markets", params={
        "series_ticker": series, "status": "open", "limit": 100
    }, timeout=15, headers={"User-Agent": "KalshiResearch-v7.1-readonly"})
    r.raise_for_status()
    return choose_market(r.json().get("markets", []))


def write(sink, kind, **data):
    sink.write(json.dumps({"type": kind, "local_ts_ms": int(time.time()*1000), **data},
                          separators=(",", ":")) + "\n")


async def capture(ticker, seconds, max_messages, log_every=10000):
    """Capture a bounded session. A reconnect begins a NEW file and snapshot."""
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / f"{ticker}_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl"
    counts = {"orderbook_snapshot": 0, "orderbook_delta": 0, "trade": 0,
              "other": 0, "sequence_gaps": 0, "delta_without_snapshot": 0}
    start = time.monotonic()
    print(f"Recording {ticker} -> {file.name}", flush=True)
    with file.open("w", encoding="utf-8", buffering=1) as sink:
        write(sink, "capture_start", ticker=ticker, version="7.1")
        async with websockets.connect(WS, additional_headers=headers(),
                                      ping_interval=20, ping_timeout=20,
                                      max_queue=8192) as ws:
            await ws.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                "channels": ["orderbook_delta", "trade"], "market_tickers": [ticker]}}))
            sequence = {}
            synced = set()
            sid_ticker = {}
            pending_resync = set()
            next_id = 100
            total = 0
            while time.monotonic()-start < seconds and total < max_messages:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                receipt = int(time.time()*1000)
                event = json.loads(raw)
                sink.write(json.dumps({"local_ts_ms": receipt, "event": event},
                                      separators=(",", ":")) + "\n")
                total += 1
                kind = event.get("type")
                key = kind if kind in ("orderbook_snapshot", "orderbook_delta", "trade") else "other"
                counts[key] += 1
                sid = event.get("sid")
                seq = event.get("seq")
                msg = event.get("msg") or {}
                if kind == "orderbook_snapshot":
                    if sid is not None:
                        synced.add(sid)
                        pending_resync.discard(sid)
                        sequence[sid] = seq
                        sid_ticker[sid] = msg.get("market_ticker", ticker)
                elif kind == "orderbook_delta":
                    if sid not in synced:
                        counts["delta_without_snapshot"] += 1
                    elif sid in sequence and seq is not None and sequence[sid] is not None:
                        if seq != sequence[sid]+1:
                            counts["sequence_gaps"] += 1
                            synced.discard(sid)
                            write(sink, "book_invalid", ticker=ticker, sid=sid,
                                  previous_seq=sequence[sid], new_seq=seq)
                    if sid not in synced and sid not in pending_resync and sid is not None:
                        # Recover by replacing the connection in the outer loop.
                        # Avoid relying on undocumented in-place snapshot commands.
                        pending_resync.add(sid)
                        write(sink, "reconnect_required", reason="book_out_of_sync")
                        print("Book unsynchronized; reconnecting for a fresh snapshot.", flush=True)
                        break
                    if sid in synced:
                        sequence[sid] = seq
                if kind == "error":
                    print("Kalshi subscription error:", msg, flush=True)
                    write(sink, "subscription_error", detail=str(msg)[:300])
                    break
                if total % log_every == 0:
                    print(f"{ticker}: {total:,} messages "
                          f"(trades={counts['trade']:,}, deltas={counts['orderbook_delta']:,})",
                          flush=True)
        write(sink, "capture_end", ticker=ticker, counts=counts)
    print(f"Session saved: {file.name} | messages={sum(counts[k] for k in ('orderbook_snapshot','orderbook_delta','trade','other')):,}",
          flush=True)
    return file, counts


async def run(series, seconds, discover_interval, max_messages):
    deadline = (time.monotonic()+seconds) if seconds else None
    recorded = 0
    sessions = 0
    last_ticker = None
    print(f"V7.1 READ-ONLY auto-rotation | {series} | Ctrl+C to stop", flush=True)
    while deadline is None or time.monotonic() < deadline:
        try:
            market = await asyncio.to_thread(discover, series)
        except (requests.RequestException, ValueError) as exc:
            print(f"Discovery failed: {exc}", flush=True)
            await asyncio.sleep(discover_interval)
            continue
        if market is None:
            print("No currently open market. Checking again.", flush=True)
            await asyncio.sleep(discover_interval)
            continue
        ticker = market["ticker"]
        remaining = (deadline-time.monotonic()) if deadline is not None else float("inf")
        market_remaining = close_epoch(market)-time.time()
        window = min(remaining, market_remaining, discover_interval, 120.0)
        if window <= .1:
            await asyncio.sleep(1)
            continue
        if ticker != last_ticker:
            print(f"Switched to {ticker}; closes {market.get('close_time')}", flush=True)
            last_ticker = ticker
        try:
            _, counts = await capture(ticker, window, max_messages)
            recorded += sum(counts.values())
            sessions += 1
        except (OSError, websockets.exceptions.WebSocketException,
                asyncio.TimeoutError, json.JSONDecodeError) as exc:
            print(f"Socket interruption ({type(exc).__name__}): {exc}", flush=True)
            await asyncio.sleep(2)
    print(f"Finished: {sessions} completed recording sessions. Output: {OUT}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--seconds",type=int,default=3600,help="0 runs until Ctrl+C")
    p.add_argument("--discovery",type=float,default=30,
                   help="Maximum seconds between active-market rediscovery")
    p.add_argument("--max-messages",type=int,default=150000,
                   help="Maximum messages per recording file/session")
    a=p.parse_args()
    if a.seconds<0 or a.discovery<5 or a.max_messages<1000:
        p.error("Require seconds>=0, discovery>=5, max-messages>=1000")
    try:
        asyncio.run(run(a.series,a.seconds,a.discovery,a.max_messages))
    except KeyboardInterrupt:
        print("Stopped; completed and buffered files remain on disk.")

if __name__=="__main__":
    main()
