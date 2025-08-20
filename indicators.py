import yfinance as yf
import pandas as pd
import numpy as np
#WILLIAM%R
'''
Every day take i.e last 14 days data
Compute:

HighestHigh = Highest price among last 14 days
LowestLow = lowest price among last 14 days
Close = current day closing price
W%R(today) = [(HighestHigh-Close)/(HighestHigh-LowestLow)] * 100
range = [-100,0]
to be optimized [buy_treshold, sell_treshold]
normal_values = Buy < -80 (oversold), Sell > -20 (overbought).
'''


def W100R_indicator(buy_treshold, sell_treshold, close, high, low):
    """
    shows trends of a stock being overbought or oversold, if the indicator is below buy_treshold,
    is oversold -> buy elif indicator is above sell_treshold is overbought -> sell
    should return a list of buy and sell signals
    """
    h_high = highest_high(14, high)
    l_low = lowest_low(14, low)
    W100R = ((h_high - close)/(h_high - l_low)) * -100
    buy_t, sell_t = buy_treshold, sell_treshold

    conditions = [
        W100R > sell_t,
        W100R < buy_t
    ]
    choices = [False, True]

    result = pd.Series(np.select(conditions, choices, default=None), index=W100R.index)
    print(result)


    return result

def highest_high(days, high):
    '''
    takes price pd.series
    return highest closing price among last 14 days
    '''
    return high.rolling(window=days).max()

def lowest_low(days, low):
    '''
    takes price pd.series
    return lowest closing price among last 14 days
    '''
    return low.rolling(window=days).min()

if __name__ == "__main__":
    stock = yf.Ticker("AAPL")
    history = stock.history(interval="1d", period="1y", start="2012-01-01")
    close = history["Close"]
    low = history["Low"]
    high = history["High"]
    williams_r = W100R_indicator(-80, -20, close, high, low)
