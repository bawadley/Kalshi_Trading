"""Offline V7 tests: snapshots, deltas, sequence gaps, and depth accounting."""
import unittest
from decimal import Decimal
from v7_replay import Book

class ReplayV7Tests(unittest.TestCase):
    def make_book(self):
        b=Book()
        b.snapshot({"yes_dollars_fp":[["0.30","15"],["0.28","20"]],
                    "no_dollars_fp":[["0.65","10"],["0.60","15"]],"clock":10},2)
        return b

    def test_spread_and_complement(self):
        b=self.make_book()
        self.assertEqual(b.top()["yes_bid"],Decimal(".30"))
        self.assertEqual(b.top()["yes_ask"],Decimal(".35"))

    def test_depth_weighted_taker(self):
        b=self.make_book()
        self.assertEqual(b.taker("buy_yes",5),Decimal(".35"))
        self.assertEqual(b.taker("sell_yes",5),Decimal(".30"))
        self.assertIsNone(b.taker("buy_yes",100))

    def test_delta_and_sequence_gap_invalidate(self):
        b=self.make_book()
        self.assertTrue(b.delta({"side":"yes","price_dollars":"0.30",
                                 "delta_fp":"-5","clock":11},3))
        self.assertEqual(b.top()["yes_bid_qty"],Decimal(10))
        self.assertFalse(b.delta({"side":"yes","price_dollars":"0.30",
                                  "delta_fp":"-1","clock":14},5))
        self.assertIsNone(b.top())

    def test_snapshot_restores_validity(self):
        b=self.make_book()
        b.delta({"side":"yes","price_dollars":"0.30","delta_fp":"-1"},9)
        self.assertFalse(b.valid)
        b.snapshot({"yes_dollars_fp":[["0.40","3"]],
                    "no_dollars_fp":[["0.50","4"]]},10)
        self.assertEqual(b.top()["yes_ask"],Decimal(".50"))

if __name__=="__main__":unittest.main()
