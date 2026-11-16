"""V7.2 uninterrupted Kalshi BTC recorder: market discovery without scheduled WS reconnects.

Read-only. Uses the V7 Ed25519/RSA authentication. Each MARKET/connection
session gets a separate JSONL file; no arbitrary reconnect at discovery ticks.
"""
import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import websockets

from v7_ws_recorder import WS, OUT, headers

REST = "https://external-api.kalshi.com/trade-api/v2"


def closing_epoch(m):
    raw=m.get("close_time") or m.get("expiration_time")
    try:
        return datetime.fromisoformat(raw.replace("Z","+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def choose_active(markets, now=None):
    now=time.time() if now is None else now
    choices=[]
    for m in markets:
        closing=closing_epoch(m)
        if (m.get("status") in ("open","active") and m.get("ticker")
                and closing is not None and closing>now):
            choices.append((closing,m["ticker"]))
    return min(choices)[1] if choices else None


def discover(series):
    r=requests.get(REST+"/markets",
                   params={"series_ticker":series,"status":"open","limit":100},
                   timeout=12,headers={"User-Agent":"KalshiResearch-v7.2"})
    r.raise_for_status()
    return choose_active(r.json().get("markets",[]))


def marker(handle,kind,**data):
    handle.write(json.dumps({"type":kind,"local_ts_ms":int(time.time()*1000),
                             **data},separators=(",",":"))+"\n")


async def discovery_loop(state,series,interval,stop):
    while not stop.is_set():
        try:
            new=await asyncio.to_thread(discover,series)
            if new != state["ticker"]:
                print(f"Market discovery: {state['ticker']} -> {new}",flush=True)
            state["ticker"]=new
            state["last_ok"]=time.monotonic()
        except (requests.RequestException,ValueError) as exc:
            print(f"Market discovery error (retaining previous choice): {exc}",flush=True)
        try:
            await asyncio.wait_for(stop.wait(),timeout=interval)
        except asyncio.TimeoutError:
            pass


async def session(ticker,state,stop,deadline,log_every):
    OUT.mkdir(parents=True,exist_ok=True)
    filename=OUT/f"{ticker}_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl"
    total=0
    seq={}
    synced=set()
    counters={"orderbook_snapshot":0,"orderbook_delta":0,"trade":0,"other":0}
    print(f"Connecting {ticker}; writing {filename.name}",flush=True)
    with filename.open("w",encoding="utf-8",buffering=1) as f:
        marker(f,"capture_start",ticker=ticker,version="7.2")
        try:
            async with websockets.connect(WS,additional_headers=headers(),
                                          ping_interval=20,ping_timeout=20,
                                          max_queue=8192) as ws:
                await ws.send(json.dumps({"id":1,"cmd":"subscribe","params":{
                    "channels":["orderbook_delta","trade"],"market_tickers":[ticker]}}))
                while (not stop.is_set() and state["ticker"]==ticker and
                       (deadline is None or time.monotonic()<deadline)):
                    try:
                        raw=await asyncio.wait_for(ws.recv(),timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    arrival=int(time.time()*1000)
                    event=json.loads(raw)
                    f.write(json.dumps({"local_ts_ms":arrival,"event":event},
                                       separators=(",",":"))+"\n")
                    total+=1
                    kind=event.get("type")
                    counters[kind if kind in counters else "other"]+=1
                    sid=event.get("sid")
                    number=event.get("seq")
                    msg=event.get("msg") or {}
                    if kind=="orderbook_snapshot":
                        if sid is not None and number is not None:
                            synced.add(sid)
                            seq[sid]=number
                        else:
                            marker(f,"book_invalid",reason="snapshot_without_sid_seq")
                            return
                    elif kind=="orderbook_delta":
                        if (sid not in synced or number is None or
                            number != seq.get(sid,-100)+1):
                            marker(f,"book_invalid",reason="delta_sequence_gap",
                                   sid=sid,previous_seq=seq.get(sid),new_seq=number)
                            print("Book sequence issue. Reconnecting for a fresh snapshot.",
                                  flush=True)
                            return
                        seq[sid]=number
                    elif kind=="error":
                        marker(f,"subscription_error",detail=str(msg)[:500])
                        print("Subscription error:",msg,flush=True)
                        return
                    if total%log_every==0:
                        print(f"{ticker}: {total:,} messages (trades={counters['trade']:,}, "
                              f"deltas={counters['orderbook_delta']:,})",flush=True)
        finally:
            marker(f,"capture_end",ticker=ticker,counters=counters)
            print(f"Saved {filename.name}: {total:,} messages",flush=True)


async def run(series,seconds,interval,log_every):
    deadline=time.monotonic()+seconds if seconds else None
    state={"ticker":None,"last_ok":None}
    stop=asyncio.Event()
    discovery=asyncio.create_task(discovery_loop(state,series,interval,stop))
    retry=1
    try:
        while deadline is None or time.monotonic()<deadline:
            ticker=state["ticker"]
            if ticker is None:
                await asyncio.sleep(1)
                continue
            try:
                await session(ticker,state,stop,deadline,log_every)
                retry=1
            except (OSError,websockets.exceptions.WebSocketException,
                    asyncio.TimeoutError,json.JSONDecodeError) as exc:
                print(f"WebSocket interrupted: {type(exc).__name__} ({str(exc)[:150]})",
                      flush=True)
                await asyncio.sleep(retry)
                retry=min(30,retry*2)
            # Preserve data even after a failure; subsequent session is a new file
            # with a new subscription and its own initial full snapshot.
            if state["ticker"]==ticker:
                await asyncio.sleep(1)
    finally:
        stop.set()
        await discovery
    print("Recording finished. Raw JSONL files retained under",OUT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--seconds",type=int,default=0,help="0: until Ctrl+C")
    p.add_argument("--discovery",type=float,default=15,
                   help="Check active market while the WS connection stays open")
    p.add_argument("--log-every",type=int,default=50000)
    a=p.parse_args()
    if a.seconds<0 or a.discovery<5 or a.log_every<100:
        p.error("Require seconds>=0, discovery>=5, log-every>=100")
    try:
        asyncio.run(run(a.series,a.seconds,a.discovery,a.log_every))
    except KeyboardInterrupt:
        print("Stopped. Previously written lines remain in recording files.")


if __name__=="__main__":
    main()
