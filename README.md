# What Bars Cannot See: Order Flow, Its OHLCV Proxies, and Short-Horizon Bitcoin Returns

Seyed Abolfazl Hosseini (Ferdowsi University of Mashhad) · hosseini.systems@gmail.com

Working paper (`paper/main.pdf`) and replication package. The study extends an Applied Statistical
Analysis course project (summer 2026). It asks three questions:

1. **Measurement.** How well do microstructure proxies computed from plain OHLCV bars (bulk volume
   classification, Corwin-Schultz and Abdi-Ranaldo spreads, Kyle's lambda, range-based volatility)
   track the quantities they claim to measure? Binance bars report aggressor-side (taker) volume, so
   the true signed order flow of every bar is observed and the proxies can be scored against it.
2. **Prediction.** Do these proxies, and their observed counterparts, forecast 4-hour returns out of
   sample, and do they add anything beyond conventional technical indicators?
3. **Economic value.** Does any statistical predictability survive realistic transaction costs?

## Main findings (BTC/USDT, Jan 2022 - Oct 2026; ETH/USDT as second asset)

1. **Volatility is measured well, order flow is not.** Garman-Klass volatility from hourly bars has
   correlation 0.99 with 1-minute realised volatility. BVC imbalance has correlation 0.49 with
   observed taker imbalance, and *no* function of the bar return explains more than 0.25 of its
   variance even in sample (information ceiling; 0.34 on 1-minute bars). BVC is already close to
   this bound and its scale drifts over time. High-low spread estimators measure volatility (median
   Corwin-Schultz 15.8 bps vs a 0.002 bps tick).
2. **In BTC, the information in flow is in the part bars cannot see.** Observed taker imbalance
   predicts a ~1.3 bps reversal over 2 hours (t = -4.0), robust to lagged returns, volume and
   volatility, stable across halves and fee regimes, unchanged when the bar return is held fixed.
   Predictive content rises with measurement accuracy: none for hourly BVC, part for 1-minute BVC,
   all for observed flow (which subsumes 1-minute BVC). Weaker and insignificant in ETH.
3. **Detectable out of sample only when measured; too small to trade.** Observed imbalance alone has
   OOS R2 ~0.02% at 2h (Clark-West p ~ 1e-4, significant in both halves) but breaks even at ~1 bp
   per unit turnover. No multivariate model beats a zero forecast; cross-validation picks the empty
   model in most months; the best of 16 trading rules has a deflated Sharpe ratio of 0.70.
4. **Fees change the data.** Binance's BTC zero-fee promotion (Jul 2022 - Mar 2023) multiplied trade
   counts ~5x and cut the dispersion of taker imbalance from 0.14 to 0.03; ETH was unaffected.

## Design choices that matter

| Issue | How it is handled |
|---|---|
| Look-ahead in features | Every feature at bar *t* uses bars ≤ *t* only; `tests/test_causality.py` replaces all later bars with noise and checks that features at *t* are unchanged |
| Label overlap across train/test | Training rows are kept only if their label *t + h* closes before the test block starts (purge); same rule for the inner validation split |
| Preprocessing leakage | Winsorisation bounds, standardisation and hyper-parameters are estimated inside each training window |
| One lucky test window | Expanding-window walk-forward with monthly re-estimation over the whole out-of-sample period, plus 6-month rolling window as robustness |
| Overlapping 4h targets | Newey-West (HAC) inference, Clark-West test for nested models, Diebold-Mariano, moving-block bootstrap for IC and Sharpe |
| "Is it just technical analysis?" | Technical-indicator control set; incremental Clark-West tests of microstructure over technicals |
| Backtest realism | Overlapping-sleeve portfolio (every forecast used, no double counting), costs of 0/2/5/10 bps per unit turnover, break-even cost |

## Feature sets

* **TI** (10): momentum 1/4/24/168h, RSI, MACD histogram, Bollinger %B, moving-average gaps, ATR.
* **MS_proxy** (23): computable from OHLCV alone. Spread/impact (Corwin-Schultz, Abdi-Ranaldo, Roll,
  Amihud), order flow via BVC (imbalance at 1/24h, acceleration, VPIN-style toxicity, Kyle's lambda),
  volatility (Garman-Klass, Rogers-Satchell, Yang-Zhang, vol ratio, vol-of-vol, square-root impact),
  price efficiency and bar structure, trading activity.
* **MS** = MS_proxy + observed taker imbalance, true Kyle's lambda, trade counts and average trade
  size (from Binance bars) + realised variance, jump share and realised skewness from 1-minute bars.

Removed from the course version: Fibonacci proximity, fractional differencing, sign entropy (weak
theoretical motivation), LSTM and C-LSTM (negative OOS R² with ~11k samples; replaced by a
regularised gradient-boosting benchmark), and the 4-class regime classifier (its trading rule lost
money after costs in every model). The useful finding of the classification track, the
microstructure-vs-technical comparison, is kept as a formal incremental test.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_binance.py --start 2022-01 --end 2026-10   # BTC + ETH, 1h + 1m bars
python scripts/download_binance.py --symbols BTCUSDT --no-1m --aggtrades 2026-09   # optional ground-truth check
python -m pytest -q tests                                     # leakage + analysis tests (8)
python run_all.py --data data/raw/BTCUSDT_1h.csv              # full run -> results/BTCUSDT_1H/
python run_all.py --data data/raw/ETHUSDT_1h.csv --quick      # second asset -> results/ETHUSDT_1H/
python paper/make_tables.py                                   # LaTeX tables + numbers.tex
cd paper && latexmk -pdf main.tex && latexmk -pdf research_summary.tex && cd ..
python scripts/make_arxiv.py                                  # self-contained arXiv bundle
```

Every number in the paper text is a macro generated by `paper/make_tables.py` from `results/`, so a
re-run updates the paper without hand-copying. `run_all.py --validation-only` regenerates only the
measurement, flow-response and single-predictor analyses (minutes instead of hours); `--quick`
skips the horizon, ablation and rolling-window blocks.

## Layout

```
config.yaml              horizons, walk-forward schedule, cost grid
run_all.py               end-to-end pipeline (--quick, --validation-only)
src/msr/data.py          loading and integrity checks (Binance, ccxt, generic CSV)
src/msr/features.py      causal feature construction and targets
src/msr/validation.py    proxies vs observed order flow, classifier variants, information ceiling, spreads
src/msr/analysis.py      flow-response regressions, conditional / sub-period / horizon / ablation analyses
src/msr/robust.py        controls, 1-minute BVC horse race, ceiling sensitivity, trading the single signal
src/msr/single.py        single-predictor out-of-sample walk-forward
src/msr/walkforward.py   purged expanding-window evaluation, ElasticNet and GBM
src/msr/inference.py     OOS R2, Clark-West, Diebold-Mariano, IC bootstrap, HAC
src/msr/backtest.py      overlapping-sleeve backtest, costs, deflated Sharpe ratio
src/msr/plots.py         figures
scripts/                 data download, synthetic data, arXiv bundle
tests/                   leakage tests and analysis tests
paper/                   main.tex + sec_*.tex, research_summary.tex, make_tables.py, tables/, figures/
```

## Outputs (per asset, in `results/<ASSET>/`)

`table_forecast_accuracy` (main result), `table_incremental` (MS over TI, observed over proxy flow),
`table_backtest`, `table_proxy_validation`, `table_conditional` (volatility and illiquidity regimes,
sessions), `table_subperiods`, `table_horizons`, `table_drop_group`, `table_rolling_window`,
`table_univariate` (in-sample, descriptive), figures in `figures/`, and `summary.json`.
