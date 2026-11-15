"""Summarize V7/V7.1 JSONL event recordings without loading them into RAM."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"data"/"v7"


def inspect(path):
    counts=Counter()
    markets=set()
    errors=Counter()
    bytes_=Path(path).stat().st_size
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            try:
                item=json.loads(line)
            except (ValueError, TypeError):
                errors["bad_json"]+=1
                continue
            event=item.get("event")
            if not isinstance(event,dict):
                k=item.get("type","marker")
                errors[k]+=1 if k in ("book_invalid","subscription_error",
                                         "disconnect","reconnect_required") else 0
                continue
            typ=event.get("type","unknown")
            counts[typ]+=1
            msg=event.get("msg")
            if isinstance(msg,dict) and msg.get("market_ticker"):
                markets.add(msg["market_ticker"])
    return counts,markets,errors,bytes_


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",default=str(ROOT))
    p.add_argument("--latest",type=int,default=0,
                   help="Inspect just N newest files; zero means all files")
    a=p.parse_args()
    files=sorted(Path(a.directory).glob("*.jsonl"),
                 key=lambda x:x.stat().st_mtime, reverse=True)
    if a.latest>0:files=files[:a.latest]
    if not files:
        print("No JSONL recordings found.")
        return
    totals=Counter(); errs=Counter(); tickers=set(); bytes_=0
    for file in files:
        c,m,e,size=inspect(file)
        totals.update(c);errs.update(e);tickers.update(m);bytes_+=size
    print(f"Files: {len(files)} | Market tickers: {len(tickers)} | Disk: {bytes_/1048576:.1f} MiB")
    print("Event counts:",dict(totals))
    print("Capture diagnostic markers:",dict(errs))
    print("Markets:",", ".join(sorted(tickers)[:15])+
          (" ..." if len(tickers)>15 else ""))
    print("Counts alone do NOT certify zero sequence gaps or genuine executable depth.")


if __name__=="__main__":
    main()
