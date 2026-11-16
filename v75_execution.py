"""V7.5 read-only, cost-aware YES/NO trading research.

Fit direct net-quote P&L models using chronologically earlier markets.
Select policy on validation only; no trade is the default. Displayed book
depth DOES NOT certify fills or profitable executable strategy.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from v74_orderflow import PRICE, FLOW, DYNAMICS, BTC

ROOT=Path(__file__).resolve().parent
FEATURE_SETS={
    "price":PRICE,
    "flow":PRICE+FLOW,
    "dynamic":PRICE+FLOW+DYNAMICS,
    "btc":PRICE+FLOW+DYNAMICS+BTC,
}
THRESHOLDS=(0.,.01,.02,.04,.08)


def taker_fee(price,contracts=1,multiplier=1.):
    """General 2026 fee: ceiling of M * 0.07 * C * P * (1-P) to $0.0001."""
    p=np.asarray(price,dtype=float)
    if np.any((p<0)|(p>1)) or contracts<1 or multiplier<0:
        raise ValueError("Invalid price/count/multiplier")
    value=np.ceil(.07*multiplier*contracts*p*(1-p)*10000-1e-9)/10000
    return float(value) if np.ndim(value)==0 else value


def net_taker(buy,sell,qty,slippage,multiplier):
    buy=np.minimum(1.,np.asarray(buy,dtype=float)+slippage)
    sell=np.maximum(0.,np.asarray(sell,dtype=float)-slippage)
    return qty*(sell-buy)-taker_fee(buy,qty,multiplier)-taker_fee(sell,qty,multiplier)


def build_labels(features,hold,latency,qty,max_age_ms,multiplier):
    """Require future entry and exit quotes plus every intervening second.

    Signal comes from end of second t, entry from end of t+latency, exit from
    end of t+latency+hold. Never cross file, snapshot, or gap boundaries.
    """
    d=features.copy()
    cols=["ticker","source_file","book_epoch","segment","second_ts",
          "yes_bid","yes_ask","yes_bid_qty","yes_ask_qty","quote_age_ms",
          "seconds_to_close"]
    missing=[c for c in cols if c not in d]
    if missing:raise ValueError(f"Missing enriched V7.4 columns: {missing}")
    for col in cols[2:]:d[col]=pd.to_numeric(d[col],errors="coerce")
    d=d.sort_values(["ticker","source_file","book_epoch","segment","second_ts"])
    duplicated=d.duplicated(["ticker","second_ts"],keep=False)
    if duplicated.any():
        print(f"Dropping {int(duplicated.sum())} ambiguous market-seconds")
        d=d[~duplicated].copy()
    g=d.groupby(["ticker","source_file","book_epoch","segment"],
                sort=False,dropna=False)
    for label,shift in (("entry",latency),("exit",latency+hold)):
        for col in ("second_ts","yes_bid","yes_ask","yes_bid_qty",
                    "yes_ask_qty","quote_age_ms"):
            d[f"{label}_{col}"]=g[col].shift(-shift)
    good=(d.entry_second_ts.eq(d.second_ts+latency)&
          d.exit_second_ts.eq(d.second_ts+latency+hold)&
          d.seconds_to_close.ge(latency+hold+2))
    for name in ("entry","exit"):
        bid=d[f"{name}_yes_bid"]
        ask=d[f"{name}_yes_ask"]
        good&=(bid.between(.001,.999)&ask.between(.001,.999)&bid.le(ask)&
               d[f"{name}_yes_bid_qty"].ge(qty)&
               d[f"{name}_yes_ask_qty"].ge(qty)&
               d[f"{name}_quote_age_ms"].le(max_age_ms))
    d=d.loc[good].copy()
    d["yes_buy"]=d.entry_yes_ask
    d["yes_sell"]=d.exit_yes_bid
    d["no_buy"]=1-d.entry_yes_bid
    d["no_sell"]=1-d.exit_yes_ask
    return d


def feature_columns(train,validation,test,name):
    cols=FEATURE_SETS[name]
    if any(c not in train for c in cols):return None
    if name=="btc" and any(not x[BTC].notna().all().all()
                           for x in (train,validation,test)):return None
    return [c for c in cols if train[c].notna().any()]


def model_candidates():
    return {
        "ridge":make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(),Ridge(alpha=30.)),
        "extra_trees":make_pipeline(SimpleImputer(strategy="median"),
                ExtraTreesRegressor(n_estimators=100,min_samples_leaf=20,
                                    random_state=19,n_jobs=-1)),
    }


def spaced(d,hold):
    start=d.groupby("ticker").second_ts.transform("min")
    return d.loc[(d.second_ts-start).mod(hold).eq(0)].copy()


def choose_trades(rows,threshold,slippage,qty,multiplier):
    """At most one YES or NO trade per market (first qualified signal)."""
    trades=[]
    for ticker,group in rows.groupby("ticker",sort=False):
        for _,r in group.sort_values("second_ts").iterrows():
            yes,no=float(r.pred_yes),float(r.pred_no)
            if max(yes,no)<=threshold:continue
            side="YES" if yes>=no else "NO"
            buy,sell=(r.yes_buy,r.yes_sell) if side=="YES" else (r.no_buy,r.no_sell)
            net=float(net_taker(buy,sell,qty,slippage,multiplier))
            trades.append({"ticker":ticker,"signal_ts":int(r.second_ts),
                           "side":side,"predicted_net":max(yes,no),
                           "entry_quote":float(buy),"exit_quote":float(sell),
                           "gross_proxy":qty*(float(sell)-float(buy)),
                           "net_proxy":net})
            break
    return pd.DataFrame(trades)


def choose_policy(train,validation,test,hold,qty,stress,multiplier,min_trades):
    """Choose only on validation P&L, requiring >=2 positive validation trades."""
    best=None
    val=spaced(validation,hold)
    if val.empty or train.empty:return None
    for feature_name in FEATURE_SETS:
        cols=feature_columns(train,val,test,feature_name)
        if not cols:continue
        for candidate in model_candidates():
            yes=model_candidates()[candidate]
            no=model_candidates()[candidate]
            yes.fit(train[cols],train.net_yes)
            no.fit(train[cols],train.net_no)
            scored=val.copy()
            scored["pred_yes"]=yes.predict(val[cols])
            scored["pred_no"]=no.predict(val[cols])
            for threshold in THRESHOLDS:
                trades=choose_trades(scored,threshold,stress,qty,multiplier)
                if len(trades)<min_trades:continue
                profit=float(trades.net_proxy.sum())
                positive=int((trades.net_proxy>0).sum())
                if profit<=0 or positive<min(2,min_trades):continue
                score=(profit,-len(trades),-threshold)
                if best is None or score>best[0]:
                    best=(score,feature_name,candidate,threshold,cols,yes,no,trades)
    return best


def market_folds(tickers,train_n,val_n,test_n):
    for pos in range(train_n+val_n,len(tickers)-test_n+1,test_n):
        yield (tickers[:pos-val_n],tickers[pos-val_n:pos],
               tickers[pos:pos+test_n])


def research(path,out,holds,qty,latency,slippages,multiplier,min_val_trades,
             train_n,val_n,test_n,max_age_ms):
    original=pd.read_csv(path)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    metrics=[]
    details=[]
    stress=max(slippages)
    for hold in holds:
        d=build_labels(original,hold,latency,qty,max_age_ms,multiplier)
        for side in ("yes","no"):
            d[f"net_{side}"]=net_taker(d[f"{side}_buy"],d[f"{side}_sell"],
                                      qty,stress,multiplier)
        tickers=d.groupby("ticker").second_ts.min().sort_values().index.tolist()
        print(f"{hold}s: {len(tickers)} markets, {len(d):,} quote-qualified observations")
        if len(tickers)<train_n+val_n+test_n:
            print("  Not enough eligible markets for complete folds")
            continue
        for fold,(tr_ids,va_ids,te_ids) in enumerate(
                market_folds(tickers,train_n,val_n,test_n),1):
            tr=d[d.ticker.isin(tr_ids)].copy()
            va=d[d.ticker.isin(va_ids)].copy()
            te=spaced(d[d.ticker.isin(te_ids)],hold)
            policy=choose_policy(tr,va,te,hold,qty,stress,multiplier,min_val_trades)
            common={"hold_s":hold,"fold":fold,"train_markets":len(tr_ids),
                    "validation_markets":len(va_ids),"test_markets":len(te_ids),
                    "test_opportunities":len(te)}
            if policy is None:
                for slip in slippages:
                    metrics.append({**common,"policy":"NO_TRADE",
                                    "test_trades":0,"net_proxy":0.,
                                    "slippage_each_leg":slip})
                print(f"  Fold {fold}: NO_TRADE (validation never cleared costs)")
                continue
            _,feat,model,threshold,cols,yes,no,val_trades=policy
            scored=te.copy()
            scored["pred_yes"]=yes.predict(te[cols])
            scored["pred_no"]=no.predict(te[cols])
            for slip in slippages:
                t=choose_trades(scored,threshold,slip,qty,multiplier)
                net=float(t.net_proxy.sum()) if len(t) else 0.
                name=f"{feat}/{model}"
                metrics.append({**common,"policy":name,"threshold":threshold,
                    "validation_trades":len(val_trades),
                    "validation_net_worst":float(val_trades.net_proxy.sum()),
                    "test_trades":len(t),"slippage_each_leg":slip,
                    "net_proxy":net})
                if len(t):
                    t["fold"]=fold
                    t["hold_s"]=hold
                    t["policy"]=name
                    t["slippage_each_leg"]=slip
                    details.append(t)
                print(f"  Fold {fold}: {name} slip={slip:.3f} "
                      f"trades={len(t)} net proxy USD {net:+.4f}")
    if not metrics:
        raise RuntimeError("No evaluable test folds; collect more distinct markets")
    pd.DataFrame(metrics).to_csv(out/"v75_folds.csv",index=False)
    trades=pd.concat(details,ignore_index=True) if details else pd.DataFrame(
        columns=["ticker","signal_ts","side","net_proxy"])
    trades.to_csv(out/"v75_quote_proxy_trades.csv",index=False)
    report=pd.DataFrame(metrics)
    print("Stress results (read-only quote proxies, NOT actual fills):")
    for (hold,slip),g in report.groupby(["hold_s","slippage_each_leg"]):
        print(f"  {hold}s, slippage={slip:.3f} each leg: "
              f"{int(g.test_trades.sum())} hypothetical trades, "
              f"net proxy USD {g.net_proxy.sum():+.4f}")
    print("Reports:",out)
    print("No actual fills. Previously explored markets are not a fresh holdout.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default=str(ROOT/"reports/v74/v74_enriched_features.csv"))
    p.add_argument("--output",default=str(ROOT/"reports/v75"))
    p.add_argument("--holds",default="5,15")
    p.add_argument("--contracts",type=int,default=1)
    p.add_argument("--latency-seconds",type=int,default=1)
    p.add_argument("--slippages",default="0,0.005,0.01")
    p.add_argument("--fee-multiplier",type=float,default=1.)
    p.add_argument("--train-markets",type=int,default=6)
    p.add_argument("--val-markets",type=int,default=2)
    p.add_argument("--test-markets",type=int,default=2)
    p.add_argument("--min-validation-trades",type=int,default=2)
    p.add_argument("--max-quote-age-ms",type=int,default=1200)
    a=p.parse_args()
    holds=[int(s) for s in a.holds.split(",")]
    slippages=sorted(float(s) for s in a.slippages.split(","))
    if (not holds or len(set(holds))!=len(holds) or
        any(h not in (5,15,30,60) for h in holds) or
        not slippages or any(s<0 or s>.1 for s in slippages) or
        min(a.contracts,a.latency_seconds,a.train_markets,a.val_markets,
            a.test_markets,a.min_validation_trades,a.max_quote_age_ms)<1 or
        a.min_validation_trades>a.val_markets or a.fee_multiplier<0):
        p.error("Invalid horizon, latency, slippage, size, fee, or folds")
    research(a.input,a.output,holds,a.contracts,a.latency_seconds,
             slippages,a.fee_multiplier,a.min_validation_trades,
             a.train_markets,a.val_markets,a.test_markets,a.max_quote_age_ms)


if __name__=="__main__":
    main()
