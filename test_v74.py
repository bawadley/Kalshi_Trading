"""Offline V7.4 tests for rolling feature and spot timestamp leakage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v74_orderflow import add_dynamics, join_external, folds, SETS


def rows():
    data=[]
    for i in range(22):
        data.append({
            "ticker":"KXBTC15M-TEST","source_file":"one.jsonl",
            "second_ts":100+i,"book_epoch":1,"segment":1,
            "mid":0.3+i*.001,"spread":.02,"yes_bid_qty":10,
            "yes_ask_qty":10,"imbalance5":i/100,
            "microprice":.3,"yes_depth5":100-i,
            "no_depth5":100+i,"book_deltas":5,
            "trade_count":2,"trade_volume":3,
            "yes_taker_volume":2,"no_taker_volume":1
        })
    return pd.DataFrame(data)


class V74Tests(unittest.TestCase):
    def test_rolling_features_known_only_from_past(self):
        d=add_dynamics(rows())
        r=d.loc[d.second_ts.eq(105)].iloc[0]
        self.assertAlmostEqual(r.mid_change_5s,.005)
        self.assertAlmostEqual(r.imbalance_change_5s,.05)
        self.assertAlmostEqual(r.yes_depth_change_5s,-5)
        self.assertAlmostEqual(r.trade_count_sum_5s,10)
        self.assertAlmostEqual(r.signed_taker_volume_5s,5)
        self.assertTrue(pd.isna(d.iloc[2].mid_change_5s))

    def test_gap_and_snapshot_reset_prevent_rolling_leak(self):
        raw=rows()
        raw.loc[raw.second_ts.ge(111),"book_epoch"]=2
        d=add_dynamics(raw)
        self.assertTrue(pd.isna(d.loc[d.second_ts.eq(112),"mid_change_5s"].iloc[0]))
        self.assertAlmostEqual(d.loc[d.second_ts.eq(116),"mid_change_5s"].iloc[0],.005)
        raw=rows().loc[lambda x:x.second_ts.ne(110)]
        d=add_dynamics(raw)
        self.assertTrue(pd.isna(d.loc[d.second_ts.eq(111),"mid_change_5s"].iloc[0]))

    def test_spot_join_waits_until_available_and_extra_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Path(tmp)/"spot.csv"
            m=Path(tmp)/"meta.csv"
            pd.DataFrame({
                "available_ts":[60,120,180],
                "spot_close":[100.,101.,102.]
            }).to_csv(s,index=False)
            pd.DataFrame({
                "ticker":["KXBTC15M-TEST"],
                "close_time":["1970-01-01T00:10:00Z"],
                "floor_strike":[100.]
            }).to_csv(m,index=False)
            inp=rows()[["ticker","second_ts"]].copy()
            joined=join_external(inp,s,m,spot_lag=30,max_spot_age=200)
            # second=100 => decision at 101, with 30-second safety delay,
            # so candle available at 60 is permitted, at 120 is not.
            self.assertAlmostEqual(joined.loc[joined.second_ts.eq(100),
                                               "spot_close"].iloc[0],100.)
            self.assertAlmostEqual(joined.loc[joined.second_ts.eq(120),
                                               "spot_close"].iloc[0],100.)
            self.assertAlmostEqual(joined.loc[joined.second_ts.eq(150),
                                               "spot_close"].iloc[0],101.)
            self.assertAlmostEqual(joined.loc[joined.second_ts.eq(150),
                                               "strike_distance_pct"].iloc[0],.01)

    def test_market_splits_have_no_overlap(self):
        names=list(range(13))
        out=list(folds(names,6,2,2))
        self.assertEqual(len(out),2)
        for tr,va,te in out:
            self.assertTrue(set(tr).isdisjoint(va))
            self.assertTrue(set(tr).isdisjoint(te))
            self.assertTrue(set(va).isdisjoint(te))

    def test_ablation_distinct(self):
        self.assertEqual(SETS["price_only"],["mid","spread"])
        self.assertIn("mid_change_5s",SETS["price_flow_dynamics"])
        self.assertIn("strike_distance_pct",SETS["price_flow_dynamics_btc"])
        self.assertNotIn("spot_ret1",SETS["price_flow_dynamics"])


if __name__=="__main__":
    unittest.main()
