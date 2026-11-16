# V7.5 — Research realistic YES/NO quote P&L, not just midpoint forecasts

V7.5 is **READ-ONLY**. It cannot authenticate to trading endpoints or submit any order.

## Run on Windows / PyCharm PowerShell

```powershell
git pull origin main
.\.venv\Scripts\python.exe -m unittest test_v75 -v
.\.venv\Scripts\python.exe v75_execution.py --holds 5,15 --contracts 1 --latency-seconds 1 --slippages 0,0.005,0.01
```

Input: `reports/v74/v74_enriched_features.csv` from V7.4.
Output: `reports/v75/v75_folds.csv` and
`reports/v75/v75_quote_proxy_trades.csv`.

If you have recorded more markets, process completed JSONL recordings with
`v72_features.py`, refresh matching data with `v74_backfill.py`, then
run `v74_orderflow.py` before invoking V7.5 again.

## What is being simulated

- The signal uses the end-of-second order-book features at second `t`.
- A hypothetical **taker buy** uses the displayed opposite ask at the **end**
  of second `t+1` (default one-second decision-to-entry delay).
- A hypothetical **taker sale** uses the displayed same-contract bid at the
  end of `t+1+hold`. This cannot depend on today's midpoint label.
- YES buy = YES ask; YES sell = YES bid.
- NO buy = 1 - YES bid; NO sell = 1 - YES ask.
- Each side requires valid, ordered, same-file/snapshot/segment book entries,
  all intervening one-second observations, at least `--contracts` displayed
  top-of-book liquidity on both legs, quote age <= 1200 ms, and sufficient
  time until market close.
- Fees on **each** taker leg use Kalshi's general July 7, 2026 schedule:
  `ceil_0.0001(0.07 × multiplier × contracts × P × (1 - P))`. The default
  multiplier is 1; verify the actual market/series multiplier. The simulator
  does **not** guarantee the displayed price is available to your order.
- Fees are calculated using slippage-adjusted simulated execution prices.
- Compare 0, 0.5 cent and 1 cent **per leg** slippage scenarios. A policy is
  trained/selected on the *worst* specified slippage. Scenario evaluation
  uses the same held-out signal times.

## Policy selection

Four competing feature groups: price only, order-flow, dynamic order-flow,
and order-flow plus BTC. Two model families (Ridge and Extra Trees)
each fit separate YES/NO net-return targets. Thresholds require a forecast
of more than 0, 1, 2, 4, or 8 cents of **net** value.

A policy is considered on validation ONLY when it selects >=2 markets,
shows a positive *total* return under the most adverse slippage scenario,
and has >=2 individually positive validation positions. Otherwise,
**NO_TRADE**, with $0 simulated P&L, is the chosen policy.

At most one simulated entry per market. Entire 15-minute markets are kept
together for chronological expanding training, adjacent validation, and
forward test blocks. A favorable test fold is reported, never used to
re-tune thresholds. No trade means no statistically supported edge.

## Critical limits and next phase

**Backtest survivorship:** Only signals for which both a future entry and a
future exit quote exist with adequate displayed size are scored. An actual
live entry may be followed by an absent bid, closed market, or delayed exit.
This selection is not observable at signal time and can overstate results.
We need future quote *availability / inability-to-liquidate* stress before
claiming paper profitability. Recording/file-end holes cannot be recovered.

- Displayed top-of-book liquidity does not establish a taker fill. Realistic
  order arrival, cancellation, latency, queue position, adverse selection,
  and partial fills remain unmodeled.
- The previously analyzed 13 markets are **NOT a fresh held-out sample**.
  Additional chronologically later markets are essential.
- Coinbase is a spot proxy, not Kalshi's CF Benchmarks settlement index.
- Maker orders may avoid the general taker fee, but there is NO maker-fill
  simulator here. Maker execution cannot be inferred just from a favorable
  bid/ask or from trade prints without order-level queue evidence. Do not
  interpret zero maker fees as free profit.
- No strategy can be called profitable based on this quote-proxy study.

Current authoritative fee sources:
- https://kalshi.com/fee-schedule
- https://kalshi.com/docs/kalshi-fee-schedule.pdf
