"""Offline tests for V7.1 market selection and event-file counting."""
import json
import tempfile
import unittest
from pathlib import Path

from v71_rolling_ws import choose_market, close_epoch
from v71_stats import inspect


class V71Tests(unittest.TestCase):
    def test_choose_earliest_open_and_skip_expired(self):
        markets=[
            {"ticker":"expired","status":"active","close_time":"2026-01-01T00:00:00Z"},
            {"ticker":"later","status":"active","close_time":"2026-12-01T00:30:00Z"},
            {"ticker":"earlier","status":"open","close_time":"2026-12-01T00:15:00Z"},
            {"ticker":"closed","status":"closed","close_time":"2026-12-01T00:01:00Z"}
        ]
        selected=choose_market(markets,now=1796083200)
        self.assertEqual(selected["ticker"],"earlier")

    def test_no_open_market(self):
        self.assertIsNone(choose_market([],now=1800000000))

    def test_parse_close_timestamp(self):
        self.assertIsNone(close_epoch({"close_time":"not-a-timestamp"}))
        self.assertIsNotNone(close_epoch({"close_time":"2026-12-01T00:15:00Z"}))

    def test_inspection_counts_events_and_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"capture.jsonl"
            events=[
                {"type":"capture_start"},
                {"local_ts_ms":1,"event":{"type":"orderbook_snapshot",
                                          "msg":{"market_ticker":"BTC1"}}},
                {"local_ts_ms":2,"event":{"type":"orderbook_delta",
                                          "msg":{"market_ticker":"BTC1"}}},
                {"local_ts_ms":3,"event":{"type":"trade",
                                          "msg":{"market_ticker":"BTC1"}}},
                {"type":"book_invalid"},
                {"type":"capture_end"}
            ]
            file.write_text("".join(json.dumps(e)+"\n" for e in events),encoding="utf-8")
            counts,markets,issues,size=inspect(file)
            self.assertEqual(counts["trade"],1)
            self.assertEqual(counts["orderbook_delta"],1)
            self.assertEqual(markets,{"BTC1"})
            self.assertEqual(issues["book_invalid"],1)
            self.assertGreater(size,0)

if __name__=="__main__":
    unittest.main()
