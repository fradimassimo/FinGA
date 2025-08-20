from deap import base, creator, tools
import yfinance as yf
import random
from indicators import macd_crossover_indicator, w100r_indicator, momentum_indicator
import pandas as pd
import numpy as np


def evaluate(individual, history):
    close = history["Close"]
    low = history["Low"]
    high = history["High"]

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

    macd = macd_crossover_indicator(
        macd_short_days,
        macd_long_days,
        signal_days,
        macd_buy_threshold,
        macd_sell_threshold,
        close,
    )
    w100r = w100r_indicator(
        w100r_days, w100r_buy_threshold, w100r_sell_threshold, close, high, low
    )
    momentum = momentum_indicator(
        momentum_days, momentum_buy_threshold, momentum_sell_threshold, close
    )

    df = pd.concat([close, macd, w100r, momentum], axis=1)
    df.columns = ["Close", "MACD", "W100R", "MOMENTUM"]

    df["true_count"] = (df[["MACD", "W100R", "MOMENTUM"]] == True).sum(axis=1)
    df["false_count"] = (df[["MACD", "W100R", "MOMENTUM"]] == False).sum(axis=1)
    conditions = [2 <= df["true_count"], 2 <= df["false_count"]]
    choices = [True, False]

    df["signal"] = pd.Series(
        np.select(conditions, choices, default=None), index=df.index
    )

    skip_days = df.index[
        max(macd_short_days, macd_long_days, signal_days, w100r_days, momentum_days)
    ]
    profits = []
    while True:
        try:
            df = df.loc[skip_days:]

            buy_idx = df.loc[df["signal"] == True].index[0]
            buy_price = df.loc[buy_idx, "Close"]

            sell_idx = (
                df.loc[buy_idx:].loc[df["signal"].loc[buy_idx:] == False].index[0]
            )
            sell_price = df.loc[sell_idx, "Close"]
            profit = float(sell_price - buy_price)
            profits.append(profit)
            skip_days = sell_idx
        except IndexError:
            break
    return (sum(profits),)


def algorithm(toolbox):
    pop = toolbox.population(n=100)
    CXPB, MUTPB, NGEN = 0.5, 0.3, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return pop


def main():
    creator.create(
        "FitnessMax", base.Fitness, weights=(1.0,)
    )  # Minimization problem -> weights = (-1.0)
    creator.create("Individual", list, fitness=creator.FitnessMax)

    IND_SIZE = 1  # NB using initRepeat = number of genes for each individual [treshold, pI1, pI2] but in the case below is number of cycles
    toolbox = base.Toolbox()

    toolbox.register("random", random.randint, 1, 50)  # n_slow
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
            toolbox.random,
        ),
        n=IND_SIZE,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    stock = yf.Ticker("AAPL")
    history = stock.history(interval="1d", period="1y", start="2012-01-01")
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=50, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, history=history)
    final_pop = algorithm(toolbox)
    print(final_pop)


if __name__ == "__main__":
    main()
