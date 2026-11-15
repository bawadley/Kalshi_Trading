"""Find currently open Kalshi market tickers in a series. Read only."""
import argparse
import requests

BASE="https://external-api.kalshi.com/trade-api/v2"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KX2YRRATE15M")
    p.add_argument("--limit",type=int,default=20)
    a=p.parse_args()
    r=requests.get(f"{BASE}/markets",
                   params={"series_ticker":a.series,"status":"open","limit":min(a.limit,100)},
                   timeout=20)
    r.raise_for_status()
    markets=r.json().get("markets",[])
    if not markets:
        print(f"No currently open markets found in {a.series}. Check series status or try later.")
        return
    for m in markets:
        print(f"{m.get('ticker')} | closes {m.get('close_time','unknown')} | "
              f"volume {m.get('volume_fp',m.get('volume','unknown'))} | {m.get('title','')}")
    print("\nUse an individual ticker above with market_depth.py collect --ticker ...")

if __name__=="__main__":
    main()
