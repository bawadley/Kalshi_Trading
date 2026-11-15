"""Read-only rolling Kalshi order-book snapshot collector for one or more market series.

Discovers active market tickers repeatedly, polls each book, and rotates when market
windows close. REST snapshots miss between-poll activity: NOT full L2 event replay.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
STORE = Path(__file__).resolve().parent / "data" / "depth"

def markets(session, series):
    """Find tradable windows; Kalshi may call them active rather than open."""
    response = session.get(BASE + "/markets",
                           params={"series_ticker":series,"limit":100},timeout=15)
    response.raise_for_status()
    items=response.json().get("markets",[])
    now=time.time()
    result=[]
    for m in items:
        if m.get("status") not in ("active","open"):
            continue
        close=m.get("close_time") or m.get("expiration_time")
        if close:
            try:
                end=datetime.fromisoformat(close.replace("Z","+00:00")).timestamp()
                if end<=now:continue
            except (ValueError,TypeError):
                pass
        result.append(m)
    return result

def grab(session,ticker):
    response=session.get(BASE+f"/markets/{ticker}/orderbook",timeout=15)
    response.raise_for_status()
    raw=response.json()
    book=raw.get("orderbook_fp",raw)
    if "yes_dollars" in book:
        yes,no=book.get("yes_dollars",[]),book.get("no_dollars",[])
    elif "yes_dollars_fp" in book:
        yes,no=book.get("yes_dollars_fp",[]),book.get("no_dollars_fp",[])
    else:
        raise ValueError("Unexpected order-book payload: "+str(list(book))[:200])
    return {"ts_ms":int(time.time()*1000),"ticker":ticker,
            "yes_bids":yes,"no_bids":no,"source":"kalshi_rest_snapshot",
            "valid":bool(yes and no)}

def run(series,seconds,interval,discovery,max_markets):
    STORE.mkdir(parents=True,exist_ok=True)
    sess=requests.Session()
    sess.headers["User-Agent"]="KalshiResearchLab-readonly/0.3"
    start=time.monotonic()
    known={}
    last_discovery=-float("inf")
    calls=0
    print("READ-ONLY | series="+",".join(series)+" | Ctrl+C to stop",flush=True)
    try:
        while seconds==0 or time.monotonic()-start<seconds:
            tick=time.monotonic()
            if tick-last_discovery>=discovery:
                last_discovery=tick
                for s in series:
                    try:
                        active=markets(sess,s)
                        print(f"{s}: {len(active)} active windows",flush=True)
                        for market in active[:max_markets]:
                            ticker=market["ticker"]
                            if ticker not in known:
                                print("Tracking",ticker,flush=True)
                            known[ticker]=market
                    except (requests.RequestException,KeyError,ValueError) as exc:
                        print(f"Discovery error {s}: {exc}",flush=True)
            for ticker,m in list(known.items()):
                closing=m.get("close_time") or m.get("expiration_time")
                if closing:
                    try:
                        close_ts=datetime.fromisoformat(closing.replace("Z","+00:00")).timestamp()
                        if close_ts<time.time():
                            del known[ticker]
                            continue
                    except (ValueError,TypeError):
                        pass
                try:
                    data=grab(sess,ticker)
                    with (STORE/f"{ticker}.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(data,separators=(",",":"))+"\n")
                    calls+=1
                except (requests.RequestException,ValueError) as exc:
                    print(f"Book unavailable {ticker}: {exc}",flush=True)
            if calls and calls%100==0:
                print(f"{calls} snapshots stored, tracking {len(known)} market(s)",flush=True)
            time.sleep(max(0.0,interval-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        print("Stopped.")
    print(f"Total collected snapshots: {calls}. Files: {STORE}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M",
                   help="Comma-separated series: KXBTC15M,KXETH15M,KX2YRRATE15M")
    p.add_argument("--seconds",type=int,default=3600,
                   help="Recording time, 0 = run until Ctrl+C")
    p.add_argument("--interval",type=float,default=5,
                   help="Minimum interval in seconds between polling cycles")
    p.add_argument("--discovery",type=float,default=15,
                   help="Seconds between market discovery calls")
    p.add_argument("--max-markets",type=int,default=2)
    a=p.parse_args()
    if a.seconds<0 or a.interval<1 or a.discovery<1 or a.max_markets<1:
        p.error("seconds>=0, interval>=1, discovery>=1, max-markets>=1 required")
    series=[s.strip().upper() for s in a.series.split(",") if s.strip()]
    if not series:p.error("At least one series required")
    run(series,a.seconds,a.interval,a.discovery,a.max_markets)

if __name__=="__main__":
    main()
