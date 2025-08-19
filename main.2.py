from deap import base, creator, tools
import yfinance as yf
import pandas as pd
import random

# PROBLEM DEFINITION
"""
Creator module allows to define/build your own problem
"""
creator.create(
    "FitnessMax", base.Fitness, weights=(1.0,)
)  # Minimization problem -> weights = (-1.0)
creator.create("Individual", list, fitness=creator.FitnessMax)

# INITIALIZING POP
"""
Using tools module, population is initialized following this hierarchy:
 (gene)->(individuals)->(pop) 
"""
IND_SIZE = 1  # NB using initRepeat = number of genes for each individual [treshold, pI1, pI2] but in the case below is number of cycles
toolbox = base.Toolbox()

# MACD encoding
"""
MACD is built using these ingredients:
fast EMA , slow EMA, signal line 
MACD = EMA_short − EMA_long
Signal = EMA(MACD, 9)
"""

toolbox.register("macd_short", random.randint, 5, 20)  # n_fast
toolbox.register("macd_long", random.randint, 21, 50)  # n_slow

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.macd_short, toolbox.macd_long),
    n=IND_SIZE,
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    df = macd_crossover_indicator(
        individual.macd_short, individual.macd_long, 9, "MSFT", "2012-01-01"
    )
    skip_days = df.index[50]
    profits = []
    while True:
        try:
            df = df.loc[skip_days:]

            buy_idx = df.loc[df["macd_crossover"]].index[0]
            buy_price = df.loc[buy_idx, "price"]

            sell_idx = df.loc[buy_idx:].loc[~df["macd_crossover"][buy_idx:]].index[0]
            sell_price = df.loc[sell_idx, "price"]
            profit = float(sell_price - buy_price)
            profits.append(profit)
            skip_days = sell_idx
        except IndexError:
            break
    return sum(profits)


def macd_crossover_indicator(
    ema_short_days, ema_long_days, signal_days, ticker, start_day
):
    """
    shows trends, if the short ema dips below the long ema the trend is going down
    if the short ema goes above the long, the trend is going up
    """
    stock = yf.Ticker(ticker)
    price = stock.history(interval="1d", period="6mo", start=start_day)["Close"]
    ema_short = ema_indicator(ema_short_days, price)
    ema_long = ema_indicator(ema_long_days, price)

    macd = ema_short - ema_long

    signal_curve = ema_indicator(signal_days, macd)
    macd_crossover = macd - signal_curve

    macd_crossover = macd_crossover > 0
    return pd.DataFrame({"macd_crossover": macd_crossover, "price": price})


def ema_indicator(days, price):
    return price.ewm(span=days, adjust=False).mean()


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    pop = toolbox.population(n=5)
    for ind in pop:
        print(ind)
