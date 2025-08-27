from deap import base, tools, creator
import random
from indicators import (
    macd_crossover_indicator,
    momentum_indicator,
    w100r_indicator,
    Signal,
)
import numpy as np
import pandas as pd


def feasible(individual):
    macd_short_days = individual[0]
    macd_long_days = individual[1]
    # signal_days = individual[2]
    macd_buy_threshold = individual[3]
    macd_sell_threshold = individual[4]

    # w100r_days = individual[5]
    w100r_buy_threshold = individual[6]
    w100r_sell_threshold = individual[7]

    # momentum_days = individual[8]
    momentum_buy_threshold = individual[9]
    momentum_sell_threshold = individual[10]

    if macd_short_days > macd_long_days:
        return False
    if macd_buy_threshold < macd_sell_threshold:
        return False
    if w100r_buy_threshold > w100r_sell_threshold:
        return False
    if momentum_buy_threshold > momentum_sell_threshold:
        return False
    return True


def mutate(individual, indpb):
    if random.random() < indpb:
        individual[0] = max(1, min(individual[0] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[1] = max(1, min(individual[1] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[2] = max(1, min(individual[2] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[3] = individual[3] + random.gauss(0, 5)
    if random.random() < indpb:
        individual[4] = individual[4] + random.gauss(0, 5)
    if random.random() < indpb:
        individual[5] = max(1, min(individual[5] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[6] = max(-100, min(individual[6] + random.gauss(0, 5), 0))
    if random.random() < indpb:
        individual[7] = max(-100, min(individual[7] + random.gauss(0, 5), 0))
    if random.random() < indpb:
        individual[8] = max(1, min(individual[8] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[9] = individual[9] + random.gauss(0, 5)
    if random.random() < indpb:
        individual[10] = individual[10] + random.gauss(0, 5)
    return (individual,)


def evaluate(individual, history):
    macd_short_days = individual[0]
    macd_long_days = individual[1]
    signal_days = individual[2]
    macd_buy_threshold = individual[3]
    macd_sell_threshold = individual[4]

    w100r_days = individual[5]
    w100r_buy_threshold = individual[6]
    w100r_sell_threshold = individual[7]

    momentum_days = individual[8]
    momentum_buy_threshold = individual[9]
    momentum_sell_threshold = individual[10]

    profits_per_stock = {}
    transactions_per_stock = {}
    for ticker in history.columns.get_level_values("Ticker"):
        profits = []
        stock_history = history[ticker]
        macd = macd_crossover_indicator(
            macd_short_days,
            macd_long_days,
            signal_days,
            macd_buy_threshold,
            macd_sell_threshold,
            stock_history["Close"],
        )
        w100r = w100r_indicator(
            w100r_days, w100r_buy_threshold, w100r_sell_threshold, stock_history
        )
        momentum = momentum_indicator(
            momentum_days,
            momentum_buy_threshold,
            momentum_sell_threshold,
            stock_history["Close"],
        )

        df = pd.concat([stock_history["Close"], macd, w100r, momentum], axis=1)
        df.columns = ["Close", "MACD", "W100R", "MOMENTUM"]

        df["buy_count"] = (df[["MACD", "W100R", "MOMENTUM"]] == Signal.BUY).sum(axis=1)
        df["sell_count"] = (df[["MACD", "W100R", "MOMENTUM"]] == Signal.SELL).sum(
            axis=1
        )
        conditions = [2 <= df["buy_count"], 2 <= df["sell_count"]]
        choices = [Signal.BUY, Signal.SELL]

        df["signal"] = pd.Series(
            np.select(conditions, choices, default=Signal.STAY), index=df.index
        )
        skip_days = df.index[
            max(macd_short_days, macd_long_days, signal_days, w100r_days, momentum_days)
        ]
        while True:
            try:
                df = df.loc[skip_days:]

                buy_idx = df.loc[df["signal"] == Signal.BUY].index[0]
                buy_price = df.loc[buy_idx, "Close"]

                sell_idx = (
                    df.loc[buy_idx:]
                    .loc[df["signal"].loc[buy_idx:] == Signal.SELL]
                    .index[0]
                )
                sell_price = df.loc[sell_idx, "Close"]
                profit = float(
                    (sell_price - buy_price) / buy_price
                )  # normalize by buy price, so stock price is accounted for
                profits.append(profit)
                skip_days = sell_idx
            except IndexError:
                # we run out of data, so we finish
                break

        profits_per_stock[ticker] = sum(profits)
        transactions_per_stock[ticker] = len(profits)
    return (sum(profits_per_stock.values()), sum(transactions_per_stock.values()))


def register_methods(toolbox, history):
    creator.create(
        "FitnessMaxMin", base.Fitness, weights=(1.0, -1.0)
    )  # Minimization problem -> weights = (-1.0)
    creator.create("Individual", list, fitness=creator.FitnessMaxMin)  # pyright: ignore[reportAttributeAccessIssue]

    toolbox.register("day_gene_init", random.randint, 1, 100)  # n_slow
    toolbox.register("threshold_gene_init", random.gauss, 0, 50)
    toolbox.register("w100r_threshold_gene_init", lambda: -100 * random.random())
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,  # pyright: ignore[reportAttributeAccessIssue]
        (
            toolbox.day_gene_init,
            toolbox.day_gene_init,
            toolbox.day_gene_init,
            toolbox.threshold_gene_init,
            toolbox.threshold_gene_init,
            toolbox.day_gene_init,
            toolbox.w100r_threshold_gene_init,
            toolbox.w100r_threshold_gene_init,
            toolbox.day_gene_init,
            toolbox.threshold_gene_init,
            toolbox.threshold_gene_init,
        ),
        n=1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("evaluate", evaluate, history=history)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (-10.0, 10.0)))
