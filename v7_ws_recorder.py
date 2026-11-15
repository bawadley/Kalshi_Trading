"""Read-only authenticated Kalshi WebSocket event recorder (BTC market).

Requires environment KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH.
Never submits orders. Store events in data/v7/<ticker>_<epoch>.jsonl.
"""
import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
OUT = Path(__file__).resolve().parent / "data" / "v7"

def headers():
    key_id=os.environ.get("KALSHI_API_KEY_ID")
    key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not key_id or not key_path:
        raise RuntimeError("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH as environment variables.")
    key=serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    ts=str(int(time.time()*1000))
    msg=(ts+"GET"+"/trade-api/ws/v2").encode()
    signed=key.sign(msg,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
    return {"KALSHI-ACCESS-KEY":key_id,"KALSHI-ACCESS-TIMESTAMP":ts,
            "KALSHI-ACCESS-SIGNATURE":base64.b64encode(signed).decode()}

async def record(ticker,seconds):
    OUT.mkdir(parents=True,exist_ok=True)
    outfile=OUT/f"{ticker}_{int(time.time())}.jsonl"
    deadline=time.monotonic()+seconds if seconds else None
    reconnect=1
    seen=0
    print(f"Read-only WebSocket capture for {ticker}. Output: {outfile}",flush=True)
    with outfile.open("a",encoding="utf-8",buffering=1) as sink:
        while deadline is None or time.monotonic()<deadline:
            try:
                async with websockets.connect(WS,additional_headers=headers(),
                                               ping_interval=20,ping_timeout=20,
                                               max_queue=4096) as ws:
                    reconnect=1
                    sink.write(json.dumps({"type":"capture_start","local_ts_ms":int(time.time()*1000)})+"\n")
                    await ws.send(json.dumps({"id":1,"cmd":"subscribe","params":{
                        "channels":["orderbook_delta","trade"],"market_tickers":[ticker]}}))
                    last_seq={}  # seq is per subscription ID; sid is assigned by server
                    synced=set()
                    next_id=10
                    while deadline is None or time.monotonic()<deadline:
                        timeout=5. if deadline is None else min(5.,max(.05,deadline-time.monotonic()))
                        try:
                            raw=await asyncio.wait_for(ws.recv(),timeout=timeout)
                        except asyncio.TimeoutError:
                            continue
                        now=int(time.time()*1000)
                        try:
                            event=json.loads(raw)
                        except json.JSONDecodeError:
                            sink.write(json.dumps({"type":"bad_json","local_ts_ms":now})+"\n")
                            continue
                        sink.write(json.dumps({"local_ts_ms":now,"event":event},
                                              separators=(",",":"))+"\n")
                        seen+=1
                        kind=event.get("type")
                        sid=event.get("sid")
                        seq=event.get("seq")
                        if kind=="orderbook_snapshot":
                            if sid is not None:synced.add(sid)
                        if kind=="orderbook_delta":
                            if sid not in synced:
                                print("Delta without initial snapshot; book not trusted.",flush=True)
                            if sid is not None and seq is not None and sid in last_seq and seq!=last_seq[sid]+1:
                                synced.discard(sid)
                                print(f"Gap detected sid={sid}: {last_seq[sid]} -> {seq}; requesting snapshot.",flush=True)
                                next_id+=1
                                await ws.send(json.dumps({"id":next_id,
                                    "cmd":"update_subscription","params":{"sids":[sid],
                                    "action":"get_snapshot","market_tickers":[ticker]}}))
                        if sid is not None and seq is not None:
                            last_seq[sid]=seq
                        if kind=="error":
                            print("WebSocket error:",event.get("msg"),flush=True)
                        if seen%200==0:print(f"Stored {seen} WS messages",flush=True)
            except (OSError,websockets.exceptions.WebSocketException) as exc:
                sink.write(json.dumps({"type":"disconnect","local_ts_ms":int(time.time()*1000),
                                       "message":str(exc)[:300]})+"\n")
                print("Disconnected; reconnecting with new snapshot:",type(exc).__name__,flush=True)
                await asyncio.sleep(reconnect)
                reconnect=min(30,reconnect*2)
    print(f"Captured {seen} raw messages; {outfile}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker",required=True,help="Individual open market ticker, not series")
    p.add_argument("--seconds",type=int,default=900,help="0 until Ctrl+C")
    a=p.parse_args()
    if a.seconds<0:p.error("--seconds must be >=0")
    try:
        asyncio.run(record(a.ticker,a.seconds))
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__=="__main__":main()
