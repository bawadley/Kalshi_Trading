"""Offline V7.5 tests: fee arithmetic, YES/NO conversion, gap guards and no-trade."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from v75_execution import (taker_fee,net_taker,build_labels,choose_trades,
                           feature_columns,market_folds,research)


def fixture(markets=1,seconds=36):
    rows=[]
    for j in range(markets):
        for i in range(seconds):
            x=.4+.001*i+.003*(j%3)
            rows.append({
                "ticker":f"TEST{j}","source_file":f"test{j}.jsonl",
                "book_epoch":1,"segment":1,"second_ts":10000+j*900+i,
                "yes_bid":x,"yes_ask":x+.02,
                "yes_bid_qty":5,"yes_ask_qty":7,
                "quote_age_ms":300,"seconds_to_close":200-i,
                "mid":x+.01,"spread":.02,"imbalance5":.1,
                "microprice":x+.01,"yes_depth5":100,"no_depth5":200,
                "trade_count":3,"trade_volume":5,"book_deltas":20,
                "yes_taker_volume":2,"no_taker_volume":1,
                "mid_change_5s":.001,"mid_change_15s":.002,
                "imbalance_change_5s":.001,"imbalance_change_15s":.002,
                "yes_depth_change_5s":1,"no_depth_change_5s":1,
                "trade_count_sum_5s":15,"trade_count_sum_15s":45,
                "signed_taker_volume_5s":5,"signed_taker_volume_15s":15,
                "mid_volatility_15s":.0002,
                "spot_ret1":.001,"spot_ret3":.003,
                "strike_distance_pct":.002,"spot_age_seconds":60})
    return pd.DataFrame(rows)


class V75Tests(unittest.TestCase):
    def test_general_taker_fee_rounding(self):
        self.assertAlmostEqual(taker_fee(.5),.0175)
        self.assertAlmostEqual(taker_fee(.5,100),1.75)
        self.assertAlmostEqual(taker_fee(.1),.0063)
        self.assertAlmostEqual(taker_fee(.1,2),.0126)
        self.assertAlmostEqual(taker_fee(.1,1,0),0)
        with self.assertRaises(ValueError):taker_fee(1.2)

    def test_yes_no_prices_and_net(self):
        d=build_labels(fixture(),5,1,1,1200,1)
        self.assertEqual(len(d),30)
        first=d.iloc[0]
        self.assertAlmostEqual(first.yes_buy,.421)
        self.assertAlmostEqual(first.yes_sell,.406)
        self.assertAlmostEqual(first.no_buy,.599)
        self.assertAlmostEqual(first.no_sell,.574)
        self.assertLess(net_taker(first.yes_buy,first.yes_sell,1,.005,1),0)

    def test_missing_seconds_snapshot_reset_stale_and_low_depth(self):
        raw=fixture()
        variants=[
            raw.loc[raw.second_ts.ne(10004)],
            raw.assign(book_epoch=np.where(raw.second_ts>=10004,2,1)),
            raw.assign(yes_ask_qty=np.where(raw.second_ts==10001,0,7)),
            raw.assign(quote_age_ms=np.where(raw.second_ts==10001,3000,300))
        ]
        for variant in variants:
            with self.subTest(case=len(variant)):
                d=build_labels(variant,5,1,1,1200,1)
                self.assertNotIn(10000,set(d.second_ts))

    def test_one_trade_per_market_and_no_trade(self):
        d=build_labels(fixture(2),5,1,1,1200,1)
        d["pred_yes"]=-.01
        d["pred_no"]=-.01
        self.assertTrue(choose_trades(d,0,0,1,1).empty)
        d["pred_yes"]=.1
        selected=choose_trades(d,0,0,1,1)
        self.assertEqual(len(selected),2)
        self.assertEqual(set(selected.side),{"YES"})

    def test_missing_btc_cannot_influence_model(self):
        d=fixture()
        d["spot_ret1"]=np.nan
        self.assertIsNone(feature_columns(d,d,d,"btc"))
        self.assertEqual(feature_columns(d,d,d,"price"),["mid","spread"])

    def test_market_split(self):
        names=list(range(13))
        split=list(market_folds(names,6,2,2))
        self.assertEqual(len(split),2)
        for tr,va,te in split:
            self.assertFalse(set(tr)&set(va))
            self.assertFalse(set(tr)&set(te))
            self.assertFalse(set(va)&set(te))

    def test_full_pipeline_refuses_losing_validation_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"features.csv"
            fixture(8).to_csv(path,index=False)
            dest=Path(directory)/"reports"
            research(path,dest,[5],1,1,[0,.01],1,2,4,2,2,1200)
            output=pd.read_csv(dest/"v75_folds.csv")
            self.assertEqual(len(output),2)
            self.assertTrue(output.policy.eq("NO_TRADE").all())
            self.assertTrue(output.net_proxy.eq(0).all())


if __name__=="__main__":
    unittest.main()
