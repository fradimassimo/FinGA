import pandas as pd
import numpy as np


def W100R_indicator(days, buy_threshold, sell_threshold, close, high, low):
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
    h_high = highest_high(days, high)
    l_low = lowest_low(days, low)
    W100R = ((h_high - close) / (h_high - l_low)) * -100

    conditions = [W100R < buy_threshold, W100R > sell_threshold]
    choices = [True, False]

    result = pd.Series(np.select(conditions, choices, default=None), index=W100R.index)
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


def mometum_indicator(momentum_days, buy_threshold, sell_threshold, close):
    momentum = close - close.shift(momentum_days)

    conditions = [momentum < buy_threshold, momentum > sell_threshold]
    choices = [True, False]

    result = pd.Series(
        np.select(conditions, choices, default=None), index=momentum.index
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
    choices = [True, False]
    result = pd.Series(
        np.select(conditions, choices, default=None), index=macd_crossover.index
    )
    return result


def ema_indicator(days, price):
    return price.ewm(span=days, adjust=False).mean()
