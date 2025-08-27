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

    w100r_buy_threshold = individual[4]
    w100r_sell_threshold = individual[5]

    macd_weight = individual[7]
    w100r_weight = individual[8]

    if macd_short_days > macd_long_days:
        return False
    if w100r_buy_threshold > w100r_sell_threshold:
        return False
    if macd_weight + w100r_weight > 1:
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
        individual[3] = max(1, min(individual[3] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[4] = max(-100, min(individual[4] + random.gauss(0, 5), 0))
    if random.random() < indpb:
        individual[5] = max(-100, min(individual[5] + random.gauss(0, 5), 0))
    if random.random() < indpb:
        individual[6] = max(1, min(individual[6] + random.randint(-2, 2), 100))
    if random.random() < indpb:
        individual[7] = random.random()
    if random.random() < indpb:
        individual[8] = random.random()

    return (individual,)


def evaluate(individual, history):
    macd_short_days = individual[0]
    macd_long_days = individual[1]
    signal_days = individual[2]

    w100r_days = individual[3]
    w100r_buy_threshold = individual[4]
    w100r_sell_threshold = individual[5]

    momentum_days = individual[6]

    macd_weight = individual[7]
    w100r_weight = individual[8]
    momentum_weight = 1 - macd_weight - w100r_weight

    profits_per_stock = {}
    transactions_per_stock = {}
    for ticker in history.columns.get_level_values("Ticker"):
        profits = []
        stock_history = history[ticker]
        macd = macd_crossover_indicator(
            macd_short_days,
            macd_long_days,
            signal_days,
            stock_history["Close"],
        )
        w100r = w100r_indicator(
            w100r_days, w100r_buy_threshold, w100r_sell_threshold, stock_history
        )
        momentum = momentum_indicator(
            momentum_days,
            stock_history["Close"],
        )

        df = pd.concat([stock_history["Close"], macd, w100r, momentum], axis=1)
        df.columns = ["Close", "MACD", "W100R", "MOMENTUM"]

        df["buy_count"] = (
            (df["MACD"] == Signal.BUY) * macd_weight
            + (df["W100R"] == Signal.BUY) * w100r_weight
            + (df["MOMENTUM"] == Signal.BUY) * momentum_weight
        )
        df["sell_count"] = (
            (df["MACD"] == Signal.SELL) * macd_weight
            + (df["W100R"] == Signal.SELL) * w100r_weight
            + (df["MOMENTUM"] == Signal.SELL) * momentum_weight
        )
        conditions = [0.5 <= df["buy_count"], 0.5 <= df["sell_count"]]
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
    toolbox.register("w100r_threshold_gene_init", lambda: -100 * random.random())
    toolbox.register("indicator_weight_init", random.random)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,  # pyright: ignore[reportAttributeAccessIssue]
        (
            toolbox.day_gene_init,
            toolbox.day_gene_init,
            toolbox.day_gene_init,
            toolbox.day_gene_init,
            toolbox.w100r_threshold_gene_init,
            toolbox.w100r_threshold_gene_init,
            toolbox.day_gene_init,
            toolbox.indicator_weight_init,
            toolbox.indicator_weight_init,
        ),
        n=1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("evaluate", evaluate, history=history)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (-10.0, 10_000.0)))
