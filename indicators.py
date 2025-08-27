import enum
import pandas as pd
import numpy as np
import yfinance as yf
import json


class Signal(enum.IntEnum):
    BUY = 1
    SELL = 2
    STAY = 0


def query_stock_exchange_history():
    with open("nasdaq.json", "r") as f:
        ticker_list = json.load(f)
    stocks = yf.Tickers(ticker_list)
    history = stocks.history(interval="1d", period="5y", start="2005-01-01")
    assert history is not None
    history = history.dropna(
        axis=1
    )  # discard stocks that have prices missing (maybe they were added later than 2014)
    history = history.swaplevel("Price", "Ticker", 1)  # pyright: ignore[reportArgumentType]
    return history


def w100r_indicator(days, buy_threshold, sell_threshold, history):
    """
    shows trends of a stock being overbought or oversold, if the indicator is below buy_treshold,
    is oversold -> buy elif indicator is above sell_treshold is overbought -> sell
    should return a list of buy and sell signals

    Every day take i.e last 14 days data
    Compute:
    HighestHigh = Highest price among last 14 days
    LowestLow = lowest price among last 14 days
    Close = current day closing price
    W%R(today) = [(HighestHigh-Close)/(HighestHigh-LowestLow)] * -100
    range = [-100,0]
    to be optimized [buy_treshold, sell_treshold]
    normal_values = Buy < -80 (oversold), Sell > -20 (overbought).
    """
    h_high = highest_high(days, history["High"])
    l_low = lowest_low(days, history["Low"])
    W100R = ((h_high - history["Close"]) / (h_high - l_low)) * -100

    conditions = [W100R < buy_threshold, W100R > sell_threshold]
    choices = [Signal.BUY, Signal.SELL]

    result = pd.Series(
        np.select(conditions, choices, default=Signal.STAY), index=W100R.index
    )
    return result


def highest_high(days, high):
    """
    takes price pd.series
    return highest closing price among last 14 days
    """
    return high.rolling(window=days).max()


def lowest_low(days, low):
    """
    takes price pd.series
    return lowest closing price among last 14 days
    """
    return low.rolling(window=days).min()


def momentum_indicator(momentum_days, buy_threshold, sell_threshold, close):
    momentum = close - close.shift(momentum_days)

    conditions = [momentum < buy_threshold, momentum > sell_threshold]
    choices = [Signal.BUY, Signal.SELL]

    result = pd.Series(
        np.select(conditions, choices, default=Signal.STAY), index=momentum.index
    )
    return result


def macd_crossover_indicator(
    ema_short_days, ema_long_days, signal_days, buy_threshold, sell_threshold, close
):
    """
    shows trends, if the short ema dips below the long ema the trend is going down
    if the short ema goes above the long, the trend is going up
    """
    ema_short = ema_indicator(ema_short_days, close)
    ema_long = ema_indicator(ema_long_days, close)

    macd = ema_short - ema_long

    signal_curve = ema_indicator(signal_days, macd)
    macd_crossover = macd - signal_curve

    conditions = [macd_crossover > buy_threshold, macd_crossover < sell_threshold]
    choices = [Signal.BUY, Signal.SELL]
    result = pd.Series(
        np.select(conditions, choices, default=Signal.STAY), index=macd_crossover.index
    )
    return result


def ema_indicator(days, price):
    return price.ewm(span=days, adjust=False).mean()
