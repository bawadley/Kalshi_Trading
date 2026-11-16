"""Offline V7.6 tests: fill assumptions, missing exits and policy selection."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v76_execution import (prepared, index_book, simulate, quote_ok,
                           choose_policy, strategy_results, audit_windows,
                           chronological_folds, run)


def sample(markets=1, seconds=65):
    rows=[]
    for j in range(markets):
        for i in range(seconds):
            bid=.4
            ask=.42
            if i==2:
                bid,ask=.38,.40
            if i==3:
                bid,ask=.37,.39
            if i==5:
                bid,ask=.46,.48
            rows.append({
                "ticker":f"M{j}","source_file":f"f{j}.jsonl",
                "book_epoch":1,"segment":1,"second_ts":10000+j*900+i,
                "yes_bid":bid,"yes_ask":ask,"yes_bid_qty":10,
                "yes_ask_qty":10,"quote_age_ms":300,
                "seconds_to_close":200-i,"mid_change_5s":.01,
                "imbalance5":.2})
    return prepared(pd.DataFrame(rows))


class V76Tests(unittest.TestCase):
    def test_quote_and_taker(self):
        d=sample()
        book=index_book(d)
        signal=next(d.itertuples(index=False))
        trade=simulate(book,signal,"YES","taker",2,1,5,1,0,1,0,1200)
        self.assertEqual(trade["status"],"QUOTE_COMPLETE")
        self.assertAlmostEqual(trade["entry_price"],.42)
        self.assertAlmostEqual(trade["exit_price"],.37)
        self.assertLess(trade["net_proxy"],0)

    def test_maker_touch_and_trade_through_are_different(self):
        d=sample()
        book=index_book(d)
        signal=next(d.itertuples(index=False))
        strict=simulate(book,signal,"YES","maker_through",2,1,5,1,
                        0,1,0,1200)
        touch=simulate(book,signal,"YES","maker_touch",2,1,5,1,
                       0,1,0,1200)
        self.assertEqual(strict["status"],"QUOTE_COMPLETE")
        self.assertEqual(touch["status"],"QUOTE_COMPLETE")
        self.assertEqual(strict["fill_second"],10003)
        self.assertEqual(touch["fill_second"],10002)
        self.assertAlmostEqual(strict["entry_price"],.4)
        self.assertGreater(strict["net_proxy"],touch["net_proxy"])

    def test_maker_no_fill_does_not_create_profit(self):
        d=sample()
        book=index_book(d)
        signal=next(d.itertuples(index=False))
        result=simulate(book,signal,"NO","maker_through",2,1,5,1,
                        0,1,0,1200)
        self.assertEqual(result["status"],"MAKER_NOT_FILLED_PROXY")
        self.assertEqual(result["net_proxy"],0)
        self.assertFalse(result["filled_proxy"])

    def test_missing_future_exit_penalized(self):
        d=sample(seconds=9)
        book=index_book(d)
        signal=next(d.itertuples(index=False))
        outcome=simulate(book,signal,"YES","taker",30,1,5,1,0,1,0,1200)
        self.assertEqual(outcome["status"],"EXIT_UNOBSERVED_LOSS_BOUND")
        self.assertTrue(outcome["unresolved"])
        self.assertLess(outcome["net_proxy"],-.42)
        # If entry cannot be seen, DO NOT exclude the signal.
        d2=d[d.second_ts.ne(10001)]
        missing=simulate(index_book(d2),signal,"YES","taker",2,1,5,1,
                         0,1,0,1200)
        self.assertEqual(missing["status"],"ENTRY_UNOBSERVED")
        self.assertEqual(missing["net_proxy"],-1)

    def test_missing_book_segment_prevents_future_exit(self):
        d=sample()
        d.loc[d.second_ts>=10003,"segment"]=2
        book=index_book(d)
        signal=next(d.itertuples(index=False))
        trade=simulate(book,signal,"YES","taker",2,1,5,1,0,1,0,1200)
        self.assertEqual(trade["status"],"EXIT_UNOBSERVED_LOSS_BOUND")

    def test_validation_keeps_no_trade_on_losing_fixture(self):
        d=sample(markets=2)
        # Keep a constant spread/price to make YES and NO taker strategies lose;
        # maker will not touch/cross either side.
        d["yes_bid"]=.4
        d["yes_ask"]=.42
        p=choose_policy(d,index_book(d),[30,60],1,5,1,.01,1,0,1200,5,2)
        self.assertIsNone(p)

    def test_chronological_split_and_coverage(self):
        d=sample(markets=13,seconds=70)
        order=d.groupby("ticker").second_ts.min().sort_values().index.tolist()
        folds=list(chronological_folds(order,6,2,2))
        self.assertEqual(len(folds),2)
        for tr,va,te in folds:
            self.assertFalse(set(tr)&set(va))
            self.assertFalse(set(tr)&set(te))
            self.assertFalse(set(va)&set(te))
        audit=audit_windows(d,index_book(d),[30,60],1,5,1200,5)
        self.assertEqual(set(audit.hold_s),{30,60})
        self.assertTrue((audit.signal_windows>=audit.taker_exit_observed).all())

    def test_complete_no_trade_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            d=sample(markets=10)
            d["yes_bid"]=.4
            d["yes_ask"]=.42
            src=Path(directory)/"input.csv"
            dest=Path(directory)/"out"
            d.to_csv(src,index=False)
            run(src,dest,[30],6,2,2,1,5,5,1,[0,.01],1,0,1200,2)
            report=pd.read_csv(dest/"v76_folds.csv")
            self.assertEqual(len(report),2)
            self.assertTrue(report.policy.eq("NO_TRADE").all())
            self.assertTrue(report.net_quote_proxy.eq(0).all())


if __name__=="__main__":
    unittest.main()
