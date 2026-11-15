# Kalshi 15-minute research lab (read-only)

Research starter for the 2-year Treasury yield 15-minute YES/NO series (`KX2YRRATE15M`).

## Setup

```bash
pip install -r requirements.txt
python kalshi_lab.py fetch --series KX2YRRATE15M --limit 300
python kalshi_lab.py test --series KX2YRRATE15M --horizon 3
```

`fetch` downloads recently settled and archived settled markets from Kalshi's public API, then their available 1-minute bid/ask candles into `data/<series>_minute.csv`. It never submits orders. Data availability and API rate limits may vary. For larger archived datasets, add pagination beyond the safety caps or obtain a bulk dataset.

`test` evaluates naive momentum, linear Ridge, and MLP neural-network regressors predicting the change in the midpoint `horizon` minutes ahead, across nonoverlapping market windows. It uses an approximate per-contract taker fee formula on both legs. It assumes hypothetical fills at the next minute's quoted ask and at the horizon's quoted bid. This is **not** a realistic execution simulator: minute quotes can be stale, bid/ask depths are unknown, fills are not guaranteed, and queuing and latency aren't modeled. It is appropriate only for screening signals, not for estimating investable returns.

The primary report also includes MAE of the forecasts and baseline net P&L, with chronological holdout by markets. A more robust next stage uses event-time L2 order-book data, order updates, and actual fills.

## Data sources
- Official API: https://docs.kalshi.com/getting_started/historical_data
- Official price candles: https://docs.kalshi.com/api-reference/market/get-market-candlesticks
- Third-party order-book archive: https://kalshieye.com/data/kalshi/kx2yrrate15m/
