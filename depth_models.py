"""Train chronological depth-based price and volatility models. Research only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

def prepare(raw, horizons):
    d=raw.sort_values(["ticker","ts_ms"]).drop_duplicates(["ticker","ts_ms"]).copy()
    d.ts_ms=pd.to_numeric(d.ts_ms)
    excluded={"ticker","ts_ms","source"}
    feats=[c for c in d if c not in excluded and pd.api.types.is_numeric_dtype(d[c])]
    if not {"mid","spread","imbalance_5"}.issubset(feats):
        raise ValueError("Use market_depth.py features to produce the input file.")
    groups=[]
    for ticker,g in d.groupby("ticker"):
        g=g.sort_values("ts_ms").copy()
        t=g.ts_ms.to_numpy()
        mids=g.mid.to_numpy()
        for h in horizons:
            target=t+h*1000
            idx=np.searchsorted(t,target)
            ok=np.where(idx<len(t))[0]
            valid=ok[(t[idx[ok]]-target[ok])<=10000]
            future=np.full(len(g),np.nan)
            future[valid]=mids[idx[valid]]
            g[f"change_{h}"]=future-mids
            g[f"volatility_{h}"]=abs(future-mids)
        groups.append(g)
    d=pd.concat(groups,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    return d.dropna(subset=[f"change_{h}" for h in horizons]),feats

def models():
    return {
        "ridge":make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),Ridge(alpha=20)),
        "forest":make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(
                  n_estimators=120,min_samples_leaf=8,random_state=42,n_jobs=-1)),
        "extra_trees":make_pipeline(SimpleImputer(strategy="median"),ExtraTreesRegressor(
                  n_estimators=120,min_samples_leaf=8,random_state=42,n_jobs=-1)),
        "neural":make_pipeline(SimpleImputer(strategy="median",add_indicator=True),StandardScaler(),
                  MLPRegressor(hidden_layer_sizes=(48,24),alpha=10,
                               early_stopping=True,max_iter=350,random_state=42))
    }

def run(path,horizons):
    d,feats=prepare(pd.read_csv(path),horizons)
    ticks=d.groupby("ticker").ts_ms.min().sort_values().index
    if len(ticks)<30:raise ValueError(f"Need 30+ distinct market windows; only {len(ticks)}. Record more.")
    a,b=int(.6*len(ticks)),int(.8*len(ticks))
    tr=d[d.ticker.isin(ticks[:a])]
    va=d[d.ticker.isin(ticks[a:b])]
    te=d[d.ticker.isin(ticks[b:])]
    if min(len(tr),len(va),len(te))<30:raise ValueError("Insufficient complete observations")
    print(f"Markets: train={a} validation={b-a} test={len(ticks)-b}; feature_count={len(feats)}")
    out=te[["ticker","ts_ms","mid","bid","ask"]].copy()
    for h in horizons:
        for typ in ("change","volatility"):
            target=f"{typ}_{h}"
            ytr,yva,yte=(x[target] for x in (tr,va,te))
            default=0 if typ=="change" else float(ytr.median())
            print(f"{typ} {h}s | baseline MAE={mean_absolute_error(yte,np.full(len(te),default)):.4f}")
            val={}
            for name,model in models().items():
                model.fit(tr[feats],ytr)
                pv=model.predict(va[feats])
                pt=model.predict(te[feats])
                if typ=="volatility":pt=np.maximum(pt,0)
                val[name]=mean_absolute_error(yva,pv)
                print(f"   {name:12} val={val[name]:.4f} test={mean_absolute_error(yte,pt):.4f}")
                out[f"{typ}_prediction_{h}s_{name}"]=pt
            print("   validation selected:",min(val,key=val.get))
            out[f"actual_{target}"]=yte.to_numpy()
    dest=Path("reports/depth_model_predictions.csv")
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest,index=False)
    print("Saved",dest,"| NO VERIFIED FILLS, no live trading.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default="reports/depth_features.csv")
    p.add_argument("--horizons",default="30,60,120,180,300")
    a=p.parse_args()
    h=sorted({int(v) for v in a.horizons.split(",")})
    if not h or min(h)<1:p.error("Horizon seconds must be positive")
    run(a.input,h)
