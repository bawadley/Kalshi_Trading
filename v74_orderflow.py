"""V7.4: leakage-aware rolling order flow + BTC spot ablation.

Exploratory midpoint forecasts ONLY. No orders, fill assumptions, or P&L claims.
The V7.2 data (one-second observations) supplies validated forward labels.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parent
PRICE=["mid","spread"]
FLOW=["imbalance5","microprice","yes_depth5","no_depth5","trade_count",
      "trade_volume","book_deltas","yes_taker_volume","no_taker_volume"]
DYNAMICS=["mid_change_5s","mid_change_15s","imbalance_change_5s",
          "imbalance_change_15s","yes_depth_change_5s","no_depth_change_5s",
          "trade_count_sum_5s","trade_count_sum_15s",
          "signed_taker_volume_5s","signed_taker_volume_15s",
          "mid_volatility_15s"]
BTC=["spot_ret1","spot_ret3","strike_distance_pct","seconds_to_close",
     "spot_age_seconds"]
SETS={
    "price_only": PRICE,
    "price_flow": PRICE+FLOW,
    "price_flow_dynamics": PRICE+FLOW+DYNAMICS,
    "price_flow_dynamics_btc": PRICE+FLOW+DYNAMICS+BTC,
}


def add_dynamics(raw):
    """All rolling calculations stay inside contiguous recording/snapshot segments."""
    d=raw.copy()
    for c in ["second_ts","mid","spread","imbalance5","yes_depth5","no_depth5",
              "trade_count","trade_volume","book_deltas",
              "yes_taker_volume","no_taker_volume","book_epoch","segment"]:
        if c in d:
            d[c]=pd.to_numeric(d[c],errors="coerce")
    required=["ticker","source_file","second_ts","book_epoch","segment"]+PRICE+FLOW
    missing=[k for k in required if k not in d]
    if missing: raise ValueError(f"Missing V7.2 feature columns: {missing}")
    d=d.dropna(subset=["ticker","source_file","second_ts"])
    d=d.sort_values(["ticker","source_file","second_ts"])
    duplicate=d.duplicated(["ticker","second_ts"])
    if duplicate.any():
        print(f"WARNING: {int(duplicate.sum())} overlapping market-seconds; excluding duplicates")
        # Exclude BOTH ambiguous records, not an arbitrary winner.
        d=d[~d.duplicated(["ticker","second_ts"],keep=False)].copy()
    # Existing V7.2 segment can be reused, but split again on any missing second.
    groups=["ticker","source_file","book_epoch","segment"]
    grp=d.groupby(groups,sort=False,dropna=False)
    prev=grp.second_ts.shift(1)
    d["_run"]=(prev.ne(d.second_ts-1)).groupby(
        [d[x] for x in groups],dropna=False).cumsum()
    segments=groups+["_run"]
    g=d.groupby(segments,sort=False,dropna=False)
    for h in (5,15):
        oldts=g.second_ts.shift(h)
        contiguous=oldts.eq(d.second_ts-h)
        for name,src in (("mid","mid"),("imbalance","imbalance5"),
                         ("yes_depth","yes_depth5"),("no_depth","no_depth5")):
            # Exact h-second history, no interpolations across pauses or resets.
            d[f"{name}_change_{h}s"]=(d[src]-g[src].shift(h)).where(contiguous)
        for target,src in (("trade_count_sum","trade_count"),
                           ("signed_taker_volume",None)):
            if src is None:
                d["_signed"]=d.yes_taker_volume-d.no_taker_volume
                src="_signed"
            d[f"{target}_{h}s"]=g[src].transform(
                lambda x:x.rolling(h,min_periods=h).sum())
            d.loc[~(g.second_ts.shift(h-1).eq(d.second_ts-(h-1))),
                  f"{target}_{h}s"]=np.nan
    d["mid_volatility_15s"]=g.mid.transform(
        lambda x:x.diff().rolling(15,min_periods=15).std())
    d.loc[~g.second_ts.shift(15).eq(d.second_ts-15),
          "mid_volatility_15s"]=np.nan
    return d.drop(columns=["_signed","_run"],errors="ignore")


def join_external(d,spot_path,meta_path,spot_lag=60,max_spot_age=180):
    """As-of join only spot candles previously completed PLUS a safety delay.

    Assumes Kalshi floor_strike/close_time are fixed contract terms. Spot is a
    Coinbase proxy, NOT the Kalshi settlement index.
    """
    spot=pd.read_csv(spot_path)
    if spot.empty:raise ValueError("No Coinbase candles")
    required=["available_ts","spot_close"]
    if any(c not in spot for c in required):
        raise ValueError(f"Coinbase file needs {required}")
    spot["available_ts"]=pd.to_numeric(spot.available_ts,errors="coerce")
    spot["spot_close"]=pd.to_numeric(spot.spot_close,errors="coerce")
    spot=spot.dropna(subset=required).sort_values("available_ts")
    spot=spot.drop_duplicates("available_ts",keep="last")
    # These lagged features depend only on candles completed before timestamp.
    spot["spot_ret1"]=spot.spot_close.pct_change(1,fill_method=None)
    spot["spot_ret3"]=spot.spot_close.pct_change(3,fill_method=None)
    d=d.copy()
    d["query_ts"]=d.second_ts+1-spot_lag
    d=pd.merge_asof(
        d.sort_values("query_ts"),
        spot[["available_ts","spot_close","spot_ret1","spot_ret3"]],
        left_on="query_ts",right_on="available_ts",
        direction="backward",tolerance=max_spot_age)
    d["spot_age_seconds"]=d.second_ts+1-d.available_ts
    # Verify the shift: the joined candle cannot have completed after the
    # conservative availability threshold.
    if d.available_ts.notna().any():
        assert (d.loc[d.available_ts.notna(),"available_ts"]<=
                d.loc[d.available_ts.notna(),"query_ts"]).all()
    meta=pd.read_csv(meta_path)
    for col in ("ticker","close_time","floor_strike"):
        if col not in meta:raise ValueError(f"Metadata missing {col}")
    meta=meta.drop_duplicates("ticker")
    meta["floor_strike"]=pd.to_numeric(meta.floor_strike,errors="coerce")
    close=pd.to_datetime(meta.close_time,utc=True,errors="coerce")
    meta["close_epoch"]=close.map(lambda x:x.timestamp() if pd.notna(x) else np.nan)
    d=d.merge(meta[["ticker","floor_strike","close_epoch"]],
              on="ticker",how="left",validate="many_to_one")
    d["seconds_to_close"]=d.close_epoch-(d.second_ts+1)
    d["strike_distance_pct"]=(d.spot_close-d.floor_strike)/d.floor_strike
    d.loc[d.floor_strike.le(0),"strike_distance_pct"]=np.nan
    return d.replace([np.inf,-np.inf],np.nan)


def enabled_feature_sets(train, validation, test):
    """Never train a feature set with absent external or rolling inputs.

    For the BTC experiment require complete verified spot and strike coverage
    in all chronological partitions; otherwise report it as unavailable.
    """
    available={}
    for name,cols in SETS.items():
        if any(col not in train for col in cols):
            continue
        if name=="price_flow_dynamics_btc":
            if any(not part[BTC].notna().all().all()
                   for part in (train,validation,test)):
                continue
        # Drop columns missing for every training row. An all-NaN feature
        # cannot inform a model and causes estimator-specific failures.
        filtered=[c for c in cols if train[c].notna().any()]
        if filtered and (name!="price_flow_dynamics_btc" or len(filtered)==len(cols)):
            available[name]=filtered
    return available


def models():
    return {
        "ridge":make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(),Ridge(alpha=25)),
        "extra_trees":make_pipeline(SimpleImputer(strategy="median"),
                    ExtraTreesRegressor(n_estimators=120,min_samples_leaf=12,
                                        random_state=42,n_jobs=-1)),
        "hist_boost":HistGradientBoostingRegressor(
                    max_iter=110,max_leaf_nodes=9,l2_regularization=10,
                    random_state=42)
    }


def folds(names,train_markets,val_markets,test_markets):
    for end in range(train_markets+val_markets,len(names)-test_markets+1,test_markets):
        yield (names[:end-val_markets], names[end-val_markets:end],
               names[end:end+test_markets])


def evaluation_sample(d,h):
    # One observation per horizon seconds within each market; avoid overlapping
    # test labels. File/epoch resets were already excluded by V7.2 labeling.
    first=d.groupby("ticker").second_ts.transform("min")
    return d.loc[(d.second_ts-first).mod(h).eq(0)].copy()


def run(input_path,spot_path,meta_path,out,horizons,train_n,val_n,test_n,
        min_market_rows,spot_lag,max_spot_age):
    d=add_dynamics(pd.read_csv(input_path))
    d=join_external(d,spot_path,meta_path,spot_lag,max_spot_age)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"v74_enriched_features.csv",index=False)
    summary=[]; preds=[]
    print(f"Enriched rows={len(d):,}; markets={d.ticker.nunique()}")
    print(f"BTC price available={d.spot_close.notna().mean():.1%}, "
          f"strike available={d.strike_distance_pct.notna().mean():.1%}")
    if not d.spot_close.notna().any():
        print("DATA GAP: Coinbase candles do not cover the recorded V7.2 seconds. "
              "Run v74_backfill.py to fetch spot for those recording dates.")
    if not d.floor_strike.notna().any():
        print("DATA GAP: V5 metadata has no matching recorded market tickers. "
              "Run v74_backfill.py to collect metadata for these markets.")
    for h in horizons:
        target=f"future_mid_change_{h}s"
        if target not in d: raise ValueError(f"Missing {target}")
        v=d.dropna(subset=[target]).copy()
        counts=v.groupby("ticker").size()
        v=v[v.ticker.isin(counts[counts>=min_market_rows].index)]
        markets=v.groupby("ticker").second_ts.min().sort_values().index.tolist()
        count=0
        for train_ids,val_ids,test_ids in folds(markets,train_n,val_n,test_n):
            count+=1
            tr=v[v.ticker.isin(train_ids)]
            va=evaluation_sample(v[v.ticker.isin(val_ids)],h)
            te=evaluation_sample(v[v.ticker.isin(test_ids)],h)
            if len(va)<5 or len(te)<5:continue
            truth=te[target].to_numpy()
            baseline=float(np.mean(abs(truth)))
            active_sets=enabled_feature_sets(tr,va,te)
            if "price_flow_dynamics_btc" not in active_sets:
                print(f"{h}s fold {count}: skipping BTC feature set: "
                      "missing or stale spot/strike/close-time features.")
            for label,cols in active_sets.items():
                # Only training-observed feature columns are eligible. The
                # held-out market is never used to choose model parameters.
                best=None
                for name,model in models().items():
                    model.fit(tr[cols],tr[target])
                    pred=model.predict(va[cols])
                    mae=float(mean_absolute_error(va[target],pred))
                    if best is None or (mae,name)<(best[0],best[1]):
                        best=(mae,name,model)
                selected=best[2]
                predicted=selected.predict(te[cols])
                test_mae=float(mean_absolute_error(truth,predicted))
                # An upper-bound-like screen, not a bid/ask execution P&L test.
                # Current spread alone is a minimum hurdle; actual trading adds
                # fees, slippage, and entry latency.
                hurdle=(te.spread.to_numpy()+.02)
                signals=(np.abs(predicted)>hurdle)
                summary.append({
                    "horizon_s":h,"fold":count,"features":label,
                    "model":best[1],"train_markets":len(train_ids),
                    "validation_markets":len(val_ids),"test_markets":len(test_ids),
                    "test_rows":len(te),"validation_mae":best[0],
                    "test_mae":test_mae,"no_change_mae":baseline,
                    "edge_screen_signals":int(signals.sum()),
                    "screen_cost_buffer":.02})
                p=te[["ticker","source_file","second_ts","mid","spread"]].copy()
                p["horizon_s"]=h
                p["fold"]=count
                p["features"]=label
                p["predicted_change"]=predicted
                p["actual_change"]=truth
                p["passes_spread_plus_2c"]=signals
                preds.append(p)
            current=[r for r in summary if r["horizon_s"]==h and r["fold"]==count]
            print(f"{h}s fold {count} test n={len(te)}, baseline={baseline:.5f}: "+
                  ", ".join(f"{r['features']}={r['test_mae']:.5f}" for r in current))
        if not count:print(f"{h}s: insufficient markets for complete forward folds")
    if not summary:raise RuntimeError("No usable folds. Collect more markets or reduce fold sizes.")
    metrics=pd.DataFrame(summary)
    metrics.to_csv(out/"v74_fold_metrics.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(out/"v74_predictions.csv",index=False)
    print("Sample-weighted mean errors:")
    for (h,k),g in metrics.groupby(["horizon_s","features"]):
        model=np.average(g.test_mae,weights=g.test_rows)
        baseline=np.average(g.no_change_mae,weights=g.test_rows)
        print(f"{h}s {k}: model={model:.5f}, baseline={baseline:.5f}, "+
              f"delta={baseline-model:+.5f}")
    print("Reports:",out)
    print("Read-only. Cost screen is NOT profitability and does NOT simulate fills.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default=str(ROOT/"data/v72/btc_1s_features.csv"))
    p.add_argument("--spot",default=str(ROOT/"data/v5/coinbase_btcusd_1m.csv"))
    p.add_argument("--metadata",default=str(ROOT/"data/v5/KXBTC15M_metadata.csv"))
    p.add_argument("--output",default=str(ROOT/"reports/v74"))
    p.add_argument("--horizons",default="5,15")
    p.add_argument("--train-markets",type=int,default=6)
    p.add_argument("--val-markets",type=int,default=2)
    p.add_argument("--test-markets",type=int,default=2)
    p.add_argument("--min-rows-per-market",type=int,default=20)
    p.add_argument("--spot-lag",type=int,default=60,
                   help="Additional seconds after complete Coinbase bar before use")
    p.add_argument("--max-spot-age",type=int,default=180)
    a=p.parse_args()
    horizons=[int(x.strip()) for x in a.horizons.split(",") if x.strip()]
    if (not horizons or len(set(horizons))!=len(horizons) or
        any(x not in (5,15,30,60) for x in horizons) or
        min(a.train_markets,a.val_markets,a.test_markets,a.min_rows_per_market)<1 or
        min(a.spot_lag,a.max_spot_age)<0):
        p.error("Invalid horizons, folds or spot availability offsets")
    run(a.input,a.spot,a.metadata,a.output,horizons,a.train_markets,
        a.val_markets,a.test_markets,a.min_rows_per_market,
        a.spot_lag,a.max_spot_age)


if __name__=="__main__":
    main()
