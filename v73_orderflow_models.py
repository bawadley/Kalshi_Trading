"""V7.3 exploratory BTC order-flow forecast ablation; no order placement.

Compare no-change to price-only and price+order-flow features. Entire 15-min
markets form chronological train/validation/test folds. Model selection uses
validation only; evaluations use non-overlapping h-second signal timestamps.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "v72" / "btc_1s_features.csv"
DEFAULT_OUT = ROOT / "reports" / "v73"
PRICE_ONLY = ["mid", "spread"]
FLOW_EXTRA = ["yes_bid_qty", "yes_ask_qty", "yes_depth5", "no_depth5",
              "imbalance5", "microprice", "book_deltas", "trade_count",
              "trade_volume", "yes_taker_volume", "no_taker_volume",
              "unknown_taker_volume", "quote_age_ms"]
SETS = {"price_only": PRICE_ONLY, "price_plus_flow": PRICE_ONLY + FLOW_EXTRA}


def candidates():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=False),
                               StandardScaler(), Ridge(alpha=25.0)),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=120, min_samples_leaf=12,
                                max_features=0.9, n_jobs=-1, random_state=42)),
        "hist_boost": HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=9, min_samples_leaf=25,
            l2_regularization=10, random_state=42)
    }


def prepare(path, horizon):
    target = f"future_mid_change_{horizon}s"
    d = pd.read_csv(path)
    needed = ["ticker", "source_file", "second_ts", "mid", "spread", target] + FLOW_EXTRA
    missing = [col for col in needed if col not in d.columns]
    if missing:
        raise ValueError(f"Missing V7.2 data columns: {missing}")
    d["second_ts"] = pd.to_numeric(d.second_ts, errors="coerce")
    for col in PRICE_ONLY + FLOW_EXTRA + [target]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["ticker", "source_file", "second_ts", "mid", "spread", target])
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.sort_values(["ticker", "second_ts", "source_file"])
    duplicates = d.duplicated(["ticker", "second_ts"])
    if duplicates.any():
        # Duplicate seconds indicate overlapping recordings; preserve an
        # unambiguous single observation per market-second.
        print(f"WARNING: dropping {int(duplicates.sum())} duplicate market-seconds")
        d = d.loc[~duplicates].copy()
    # Prevent a large market with more recordings from dominating all fits.
    # Comparisons remain within the SAME set of chronologically split markets.
    d = d.rename(columns={target: "target"})
    return d.reset_index(drop=True)


def markets_in_order(d):
    return (d.groupby("ticker").second_ts.min().sort_values(kind="stable")
            .index.tolist())


def split_folds(names, train_min, val_count, test_count):
    """Yield expanding train, adjacent validation, then untouched forward test."""
    if len(names) < train_min + val_count + test_count:
        return
    for start in range(train_min + val_count, len(names), test_count):
        train = names[:start - val_count]
        validation = names[start - val_count:start]
        test = names[start:start + test_count]
        if len(test) < test_count:
            break
        yield train, validation, test


def spaced_indices(d, horizon):
    """Evaluate signals only every horizon seconds within each ticker.

    Rows with intervening file boundaries are already unlabeled by V7.2.
    """
    if d.empty:
        return np.array([], dtype=bool)
    starts = d.groupby("ticker").second_ts.transform("min")
    return ((d.second_ts - starts) % horizon == 0).to_numpy(dtype=bool)


def safe_mae(y, pred):
    return float(mean_absolute_error(np.asarray(y), np.asarray(pred)))


def selected_model(train, validation, columns, horizon):
    x_train = train[columns]
    y_train = train.target
    mask = spaced_indices(validation, horizon)
    if not mask.any():
        raise ValueError("Validation has no sampled rows")
    results = []
    for name, model in candidates().items():
        model.fit(x_train, y_train)
        predicted = model.predict(validation[columns])
        loss = safe_mae(validation.target.to_numpy()[mask], predicted[mask])
        results.append((loss, name, model))
    results.sort(key=lambda r: (r[0], r[1]))
    return results[0], {name: float(loss) for loss, name, _ in results}


def run(path, output, horizons, train_min, val_count, test_count,
        min_rows_per_market):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    predictions = []
    for horizon in horizons:
        d = prepare(path, horizon)
        counts = d.groupby("ticker").size()
        eligible = counts[counts >= min_rows_per_market].index
        d = d[d.ticker.isin(eligible)].copy()
        names = markets_in_order(d)
        print(f"{horizon}s: usable markets={len(names)}, labeled rows={len(d):,}")
        folds = list(split_folds(names, train_min, val_count, test_count))
        if not folds:
            print(f"{horizon}s: not enough markets for train={train_min}, "
                  f"validation={val_count}, test={test_count}; skipping")
            continue
        for fold_id, (train_names, val_names, test_names) in enumerate(folds, 1):
            train = d[d.ticker.isin(train_names)]
            validation = d[d.ticker.isin(val_names)]
            test = d[d.ticker.isin(test_names)]
            sample = spaced_indices(test, horizon)
            if not sample.any():
                continue
            truth = test.target.to_numpy()[sample]
            naive = np.zeros(len(truth))
            naive_mae = safe_mae(truth, naive)
            for feature_set, columns in SETS.items():
                (validation_mae, winner, model), alternatives = selected_model(
                    train, validation, columns, horizon)
                predicted = model.predict(test[columns])[sample]
                loss = safe_mae(truth, predicted)
                row = {
                    "horizon_s": horizon, "fold": fold_id, "feature_set": feature_set,
                    "model": winner, "train_markets": len(train_names),
                    "validation_markets": len(val_names),
                    "test_markets": len(test_names), "test_rows": len(truth),
                    "train_rows": len(train), "validation_rows": len(validation),
                    "validation_mae": validation_mae,
                    "test_mae": loss, "baseline_mae": naive_mae,
                    "improvement_pct": (100 * (naive_mae-loss)/naive_mae
                                        if naive_mae else np.nan),
                    "validation_candidates": str({k: round(v, 6)
                                                   for k, v in alternatives.items()})
                }
                metrics.append(row)
                print(f"{horizon}s fold {fold_id} {feature_set}: {winner} | "
                      f"test MAE {loss:.5f} vs baseline {naive_mae:.5f}, "
                      f"n={len(truth)}, improvement={row['improvement_pct']:+.1f}%")
                t = test.loc[sample, ["ticker", "source_file", "second_ts"]].copy()
                t["horizon_s"] = horizon
                t["fold"] = fold_id
                t["feature_set"] = feature_set
                t["selected_model"] = winner
                t["actual_change"] = truth
                t["predicted_change"] = predicted
                predictions.append(t)
    if not metrics:
        raise RuntimeError("No evaluable folds. Collect more distinct complete markets "
                           "or lower --train-markets / --val-markets / --test-markets "
                           "for a software diagnostic.")
    frame = pd.DataFrame(metrics)
    frame.to_csv(output/"v73_fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output/"v73_predictions.csv", index=False)
    print("\nWeighted summary (sample-size-weighted within each feature set):")
    for (horizon, feature), group in frame.groupby(["horizon_s", "feature_set"]):
        actual = np.average(group.test_mae, weights=group.test_rows)
        reference = np.average(group.baseline_mae, weights=group.test_rows)
        print(f"{horizon}s {feature}: model={actual:.5f}, no-change={reference:.5f}, "
              f"delta={reference-actual:+.5f}, folds={len(group)}")
    print("Reports:", output)
    print("EXPLORATORY FORECAST RESULTS ONLY. 13 markets is a very small independent "
          "sample; overlapping training targets and receive-time latency still matter.")
    print("No trades placed; no executable profitability estimated.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--horizons", default="5,15")
    p.add_argument("--train-markets", type=int, default=6)
    p.add_argument("--val-markets", type=int, default=2)
    p.add_argument("--test-markets", type=int, default=2)
    p.add_argument("--min-rows-per-market", type=int, default=20)
    a = p.parse_args()
    horizons = [int(s.strip()) for s in a.horizons.split(",") if s.strip()]
    if (not horizons or any(h not in (5, 15, 30, 60) for h in horizons)
        or len(set(horizons)) != len(horizons)
        or min(a.train_markets, a.val_markets, a.test_markets,
               a.min_rows_per_market) < 1):
        p.error("Use unique horizons from 5,15,30,60 and positive market/row counts.")
    run(a.input, a.output, horizons, a.train_markets, a.val_markets,
        a.test_markets, a.min_rows_per_market)


if __name__ == "__main__":
    main()
