"""V5: timestamp-safe BTC spot features and expanding walk-forward price evaluation.

Read-only research. Spot source is NOT Kalshi settlement index.
No live orders. Candle quotes are not executable fills.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor

ROOT=Path(__file__).resolve().parent
BASE_FEATURES=["mid","spread","age","r1","r2","r3","volume","roll_vol3",
               "spot_close","spot_ret1","spot_ret3","spot_range","spot_volume",
               "minutes_to_close","spot_age_seconds"]
OPTIONAL=["strike_distance_pct"]

def read(series,horizon,max_spot_age):
    source=ROOT/"data"/f"{series}_minute.csv"
    metadata=ROOT/"data"/"v5"/f"{series}_metadata.csv"
    spot_file=ROOT/"data"/"v5"/"coinbase_btcusd_1m.csv"
    for path in (source,metadata,spot_file):
        if not path.exists():raise FileNotFoundError(f"Missing {path}. Run btc_v5_data.py first.")
    quotes=pd.read_csv(source)
    for col in ("end_period_ts","bid","ask","volume"):
        quotes[col]=pd.to_numeric(quotes[col],errors="coerce")
    quotes=quotes.dropna(subset=["ticker","end_period_ts","bid","ask"])
    quotes=quotes[(quotes.bid>=0)&(quotes.ask<=1)&(quotes.bid<=quotes.ask)]
    quotes=quotes.sort_values(["ticker","end_period_ts"]).drop_duplicates(["ticker","end_period_ts"])
    quotes["mid"]=(quotes.bid+quotes.ask)/2
    quotes["spread"]=quotes.ask-quotes.bid
    g=quotes.groupby("ticker",sort=False)
    quotes["age"]=g.cumcount()
    for lag in (1,2,3):
        quotes[f"r{lag}"]=quotes.mid-g.mid.shift(lag)
    quotes["roll_vol3"]=g.mid.transform(lambda x:x.diff().rolling(3).std())
    quotes["future_mid"]=g.mid.shift(-horizon)
    quotes["future_ts"]=g.end_period_ts.shift(-horizon)
    quotes["target"]=quotes.future_mid-quotes.mid
    # Reject gaps in the observed candle timeline.
    quotes=quotes[(quotes.future_ts-quotes.end_period_ts)==60*horizon].copy()

    m=pd.read_csv(metadata)
    if m.empty:raise RuntimeError("No Kalshi market metadata available")
    m=m.drop_duplicates("ticker")
    quotes=quotes.merge(m[["ticker","close_time","expiration_time","floor_strike"]],
                        on="ticker",how="left")
    c=pd.to_datetime(quotes.close_time.fillna(quotes.expiration_time),utc=True,errors="coerce")
    quotes["minutes_to_close"]=(c.astype("int64")/1e9-quotes.end_period_ts)/60
    quotes.loc[c.isna(),"minutes_to_close"]=np.nan
    quotes["floor_strike"]=pd.to_numeric(quotes.floor_strike,errors="coerce")

    spot=pd.read_csv(spot_file)
    if spot.empty:raise RuntimeError("Spot file empty")
    for c in ("available_ts","spot_close","spot_open","spot_low","spot_high","spot_volume"):
        spot[c]=pd.to_numeric(spot[c],errors="coerce")
    spot=spot.dropna(subset=["available_ts","spot_close"]).sort_values("available_ts")
    # Spot bar CLOSE is only known once its minute is complete.
    spot["spot_ret1"]=spot.spot_close.pct_change()
    spot["spot_ret3"]=spot.spot_close.pct_change(3)
    spot["spot_range"]=(spot.spot_high-spot.spot_low)/spot.spot_close
    spot=spot.drop_duplicates("available_ts")
    z=pd.merge_asof(quotes.sort_values("end_period_ts"),
                    spot[["available_ts","spot_close","spot_ret1","spot_ret3",
                          "spot_range","spot_volume"]].sort_values("available_ts"),
                    left_on="end_period_ts",right_on="available_ts",
                    direction="backward",tolerance=max_spot_age)
    z["spot_age_seconds"]=z.end_period_ts-z.available_ts
    z["strike_distance_pct"]=(z.spot_close-z.floor_strike)/z.floor_strike
    # Do not impute missing strike with an inferred or future threshold.
    z=z.replace([np.inf,-np.inf],np.nan)
    return z

def walk_forward(z,horizon,min_train_markets,fold_markets):
    valid=z.dropna(subset=BASE_FEATURES+["target"])
    windows=valid.groupby("ticker").end_period_ts.min().sort_values()
    tickers=windows.index.to_list()
    n=len(tickers)
    if n<min_train_markets+fold_markets:
        raise RuntimeError(f"Need {min_train_markets+fold_markets} usable market windows; have {n}")
    print(f"Market windows={n}; rows={len(valid)}; strike metadata present="
          f"{valid.strike_distance_pct.notna().mean():.1%}")
    print("Coinbase is a proxy; not the settlement index. All features are known at signal time.")
    result=[]
    for end in range(min_train_markets,n,fold_markets):
        test_names=tickers[end:min(n,end+fold_markets)]
        if len(test_names)<max(10,fold_markets//2):break
        train=valid[valid.ticker.isin(tickers[:end])]
        test=valid[valid.ticker.isin(test_names)]
        # Optional strike feature is only enabled when ALL training and holdout
        # observations have verified numeric strike metadata.
        feat=list(BASE_FEATURES)
        if train.strike_distance_pct.notna().all() and test.strike_distance_pct.notna().all():
            feat+=OPTIONAL
        model=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=9,
                                             l2_regularization=8,random_state=42)
        model.fit(train[feat],train.target)
        prediction=model.predict(test[feat])
        truth=test.target.to_numpy()
        baseline=np.zeros(len(test))
        row={"fold":len(result)+1,"train_markets":end,"test_markets":len(test_names),
             "test_rows":len(test),"features":len(feat),
             "model_mae":mean_absolute_error(truth,prediction),
             "baseline_mae":mean_absolute_error(truth,baseline)}
        result.append(row)
        print(f"Fold {row['fold']}: train={end}, test={len(test_names)}; "
              f"MAE model={row['model_mae']:.4f} baseline={row['baseline_mae']:.4f}; "
              f"features={len(feat)}")
    if not result:raise RuntimeError("No complete forward test folds")
    report=pd.DataFrame(result)
    dest=ROOT/"reports"/f"{series_name(z)}_v5_walkforward.csv"
    dest.parent.mkdir(exist_ok=True)
    report.to_csv(dest,index=False)
    weighted=np.average(report.model_mae,weights=report.test_rows)
    weighted_base=np.average(report.baseline_mae,weights=report.test_rows)
    print(f"Weighted MAE: model={weighted:.4f}, baseline={weighted_base:.4f}")
    print("Saved",dest)
    print("No trade P&L reported: this evaluates price forecast quality only.")

def series_name(d):
    # Derived from known Kalshi ticker prefix; only used for report naming.
    return str(d.ticker.iloc[0]).split("-")[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series",default="KXBTC15M")
    p.add_argument("--horizon",type=int,default=2)
    p.add_argument("--min-train-markets",type=int,default=150)
    p.add_argument("--fold-markets",type=int,default=50)
    p.add_argument("--max-spot-age",type=int,default=120)
    a=p.parse_args()
    if not 1<=a.horizon<=8 or a.min_train_markets<50 or a.fold_markets<10 or a.max_spot_age<0:
        p.error("Invalid horizon, training, fold, or spot-age value.")
    z=read(a.series,a.horizon,a.max_spot_age)
    walk_forward(z,a.horizon,a.min_train_markets,a.fold_markets)

if __name__=="__main__":main()
