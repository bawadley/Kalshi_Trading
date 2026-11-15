"""Lightweight tests for execution alignment and order-fee calculations."""
import unittest
import numpy as np
import pandas as pd
from btc_execution_v3 import taker_fee, evaluate

class ExecutionTests(unittest.TestCase):
    def test_fees_positive_and_round(self):
        self.assertAlmostEqual(float(taker_fee(.5,100)),1.75)
        self.assertAlmostEqual(float(taker_fee(.5,1)),.0175)

    def test_one_trade_per_market_and_separate_entry_exit(self):
        base={"ticker":"T1","end_period_ts":1000,"entry_ts":1060,"exit_ts":1120,
              "pred":.20,"spread":.02,
              "entry_yes_ask":.30,"entry_yes_bid":.28,
              "exit_yes_ask":.51,"exit_yes_bid":.49}
        a=pd.DataFrame([base,dict(base,end_period_ts=1060,entry_ts=1120,exit_ts=1180)])
        trades=evaluate("test",a,np.array([.2,.2]),1,.06,1,0,1)
        self.assertEqual(len(trades),1)
        self.assertEqual(trades.iloc[0].side,"YES")
        self.assertGreater(trades.iloc[0].exit_ts,trades.iloc[0].entry_ts)

    def test_no_side_price_conversion(self):
        a=pd.DataFrame([{"ticker":"T2","end_period_ts":1000,"entry_ts":1060,
             "exit_ts":1180,"pred":-.25,"spread":.02,"entry_yes_ask":.72,
             "entry_yes_bid":.70,"exit_yes_ask":.45,"exit_yes_bid":.43}])
        trades=evaluate("test",a,np.array([-.25]),2,.06,1,0,1)
        self.assertEqual(trades.iloc[0].side,"NO")
        self.assertAlmostEqual(trades.iloc[0].buy_ask,.30)
        self.assertAlmostEqual(trades.iloc[0].sell_bid,.55)

if __name__=="__main__":
    unittest.main()
