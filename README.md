# What Bars Cannot See: Order Flow, Fee-Free Volume, and OHLCV Microstructure Proxies

Seyed Abolfazl Hosseini (Ferdowsi University of Mashhad) · hosseini.systems@gmail.com

Working paper (`paper/main.pdf`), two-page summary (`paper/research_summary.pdf`) and replication
package. Binance bars report the aggressor-side (taker) volume of every bar, so the true signed order
flow is observed. The paper uses this to ask three questions:

1. **Measurement.** How well do microstructure measures computed from plain OHLCV bars (bulk volume
   classification, Kyle's lambda, Amihud, VPIN, Corwin-Schultz and Abdi-Ranaldo spreads,
   range-based volatility) track the quantities they claim to measure? Twenty coins, 2022-2026.
2. **Volume vs. liquidity.** Binance charged no fees on bitcoin pairs from 2022-07-08 14:00 UTC to
   2023-03-22 00:00 UTC. With 19 USDT pairs that kept their fees as controls, what did fee-free
   volume consist of, and how did bar-based liquidity and toxicity measures read it?
3. **Information.** Does the part of order flow that bars cannot see predict returns, out of sample
   and after costs?

## Main findings

1. **Bars see volatility, not order flow.** Garman-Klass volatility from hourly bars has correlation
   0.93-0.99 with one-minute realised volatility in every coin. But no function of the hourly bar's
   own return explains more than about a quarter of the variance of observed taker imbalance in any
   coin (information ceiling; 0.25 for BTC, lower for thinly traded coins), and BVC is close to this
   bound. High-low spread estimators measure volatility (median Corwin-Schultz 15.8 bps for BTC vs a
   0.002 bps tick).
2. **Volume is not liquidity (natural experiment).** Removing fees multiplied BTC/USDT trades about
   sixfold relative to the controls, with breaks in the exact hour of each switch. About 93% of the
   extra volume was two-sided (offset by opposite aggressor flow within the same minute); in trade
   data the share offset within the same second doubled. The Amihud ratio and BVC-based Kyle's lambda
   fell sharply, as if liquidity had improved, while the price impact of a dollar of observed net
   flow did not fall, and VPIN computed with BVC barely moved while observed VPIN collapsed. The same
   balanced volume moved to BTC/TUSD and then BTC/FDUSD as the promotion moved. Inference uses
   permutation tests across coins and over time (one treated unit).
3. **The hidden flow is the informative flow, but it is small.** In BTC a one-SD taker-imbalance shock
   is followed by a ~1.3 bp reversal over two hours (t = -4.0), unchanged when the bar return is held
   fixed; hourly BVC predicts nothing. Across twenty coins the response is negative for most coins
   and the pooled estimate is significant with Driscoll-Kraay errors; it is driven by market-wide
   flow. Observed imbalance alone is significant out of sample for BTC but breaks even at ~1 bp, and
   no multivariate model beats a zero forecast over 53 leakage-controlled test months.

Exact numbers are in the paper; every number in the text is generated from `results/`.

## Design choices that matter

| Issue | How it is handled |
|---|---|
| Look-ahead in features | Every feature at bar *t* uses bars ≤ *t* only; `tests/test_causality.py` replaces all later bars with noise and checks that features at *t* are unchanged |
| Label overlap across train/test | Training rows are kept only if their label *t + h* closes before the test block starts (purge); same rule for the inner validation split |
| Preprocessing leakage | Winsorisation bounds, standardisation and hyper-parameters are estimated inside each training window |
| One treated unit in the fee experiment | Event-window difference-in-differences with Newey-West errors, in-space placebos (each control as if treated) and in-time placebos (weekly dates in 2024-2026); the restoration of fees as a second test |
| Correlated coins | Driscoll-Kraay standard errors for pooled panel regressions |
| Overlapping targets | Newey-West (HAC) inference, Clark-West test for nested models, Diebold-Mariano, moving-block bootstrap |
| Backtest realism | Overlapping-sleeve portfolio, costs of 0/2/5/10 bps per unit turnover, break-even cost, deflated Sharpe ratio |

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_binance.py --start 2022-01 --end 2026-10            # BTC + ETH, 1h + 1m bars
python scripts/download_binance.py --symbols BTCUSDT --no-1m --aggtrades 2026-09   # ground-truth check
python scripts/download_binance.py --outdir data/raw/panel --symbols BTCUSDT ETHUSDT BNBUSDT XRPUSDT \
    ADAUSDT SOLUSDT DOGEUSDT LTCUSDT TRXUSDT LINKUSDT DOTUSDT AVAXUSDT BCHUSDT ATOMUSDT ETCUSDT XLMUSDT \
    UNIUSDT FILUSDT NEARUSDT ALGOUSDT                                      # 20-coin panel (~2 GB)
python scripts/download_binance.py --outdir data/raw/pairs --no-1m --symbols BTCBUSD BTCTUSD BTCFDUSD BTCUSDC
python -m pytest -q tests                                     # leakage, analysis and fee-experiment tests
python run_all.py --data data/raw/BTCUSDT_1h.csv              # full run -> results/BTCUSDT_1H/
python run_all.py --data data/raw/ETHUSDT_1h.csv --quick      # second asset -> results/ETHUSDT_1H/
python run_panel.py                                           # twenty coins -> results/PANEL/
python run_fee.py                                             # zero-fee experiment -> results/FEE/
python scripts/trade_forensics.py                             # trade-level data around the switches (~0.5 GB)
python paper/make_tables.py                                   # LaTeX tables + numbers.tex
cd paper && latexmk -pdf main.tex && latexmk -pdf research_summary.tex && cd ..
python scripts/make_arxiv.py                                  # self-contained arXiv bundle
```

Every number in the paper text is a macro generated by `paper/make_tables.py` from `results/`, so a
re-run updates the paper without hand-copying. `run_all.py --validation-only` regenerates only the
measurement, flow-response and single-predictor analyses; `--quick` skips the horizon, ablation and
rolling-window blocks.

## Layout

```
config.yaml              horizons, walk-forward schedule, cost grid
run_all.py               BTC / ETH pipeline: validation, flow response, walk-forward, backtests
run_panel.py             twenty-coin measurement and flow-response panel
run_fee.py               zero-fee natural experiment
src/msr/data.py          loading and integrity checks (Binance, ccxt, generic CSV)
src/msr/features.py      causal feature construction and targets
src/msr/validation.py    proxies vs observed order flow, classifier variants, information ceiling, spreads
src/msr/events.py        fee-event timestamps
src/msr/fee.py           daily measures from one-minute bars, event-window DiD, placebos, pair migration
src/msr/analysis.py      flow-response regressions, conditional / sub-period / horizon / ablation analyses
src/msr/robust.py        controls, 1-minute BVC horse race, ceiling sensitivity, trading the single signal
src/msr/single.py        single-predictor out-of-sample walk-forward
src/msr/walkforward.py   purged expanding-window evaluation, ElasticNet and GBM
src/msr/inference.py     OOS R2, Clark-West, Diebold-Mariano, IC bootstrap, HAC
src/msr/backtest.py      overlapping-sleeve backtest, costs, deflated Sharpe ratio
src/msr/plots.py         figures
scripts/                 data download, trade-level forensics, synthetic data, arXiv bundle
tests/                   leakage tests, analysis tests, fee-experiment tests
paper/                   main.tex + sec_*.tex, research_summary.tex, make_tables.py, tables/, figures/
```
