"""Fee events on Binance spot BTC pairs (all times UTC).

Sources: Binance announcements of 6 July 2022 and 15 March 2023 as reported by CoinDesk
(2022-07-06, 2023-03-22), Decrypt (2022-07-08) and crypto.news/Kaiko (2023-09-12). The exact
hours are identified in the hourly data: trade counts on BTC/USDT jump between the 13:00 and
14:00 bars of 8 July 2022 and fall at the 00:00 bar of 22 March 2023, and nothing happens on
ETH/USDT at either hour (see fee.switch_window).
"""
import pandas as pd

# Zero maker and taker fees on 13 BTC spot pairs (BTC/USDT, BTC/BUSD, BTC/TUSD, BTC/USDC, ...).
ZERO_FEE_ON = pd.Timestamp("2022-07-08 14:00", tz="UTC")
# Standard fees again on BTC/USDT, BTC/BUSD and the fiat pairs; BTC/TUSD keeps zero fees.
ZERO_FEE_OFF = pd.Timestamp("2023-03-22 00:00", tz="UTC")
# Zero fees removed on BTC/TUSD (Kaiko, via crypto.news); BTC/FDUSD is then the only fee-free BTC pair.
TUSD_OFF = pd.Timestamp("2023-09-07 00:00", tz="UTC")


def zero_fee_mask(index: pd.DatetimeIndex):
    """True for bars of BTC/USDT traded without fees."""
    return (index >= ZERO_FEE_ON) & (index < ZERO_FEE_OFF)
