import json
from deap import base, creator, tools, algorithms
import yfinance as yf
import random
from indicators import macd_crossover_indicator, w100r_indicator, momentum_indicator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    plt.show()
    plt.savefig("result.jpg")


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
        while True:
            try:
                df = df.loc[skip_days:]

                buy_idx = df.loc[df["signal"] == True].index[0]
                buy_price = df.loc[buy_idx, "Close"]

                sell_idx = (
                    df.loc[buy_idx:].loc[df["signal"].loc[buy_idx:] == False].index[0]
                )
                sell_price = df.loc[sell_idx, "Close"]
                profit = float(
                    (sell_price - buy_price) / buy_price
                )  # normalize by buy price, so stock price is accounted for
                profits.append(profit)
                skip_days = sell_idx
            except IndexError:
                break

        profits_per_stock[ticker] = sum(profits)
        transactions_per_stock[ticker] = len(profits)
    return (sum(profits_per_stock.values()), sum(transactions_per_stock.values()))


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


def uniform_random_scaled(scaler):
    return scaler * random.random()


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


def main():
    creator.create(
        "FitnessMaxMin", base.Fitness, weights=(1.0, -1.0)
    )  # Minimization problem -> weights = (-1.0)
    creator.create("Individual", list, fitness=creator.FitnessMaxMin)

    toolbox = base.Toolbox()

    toolbox.register("day_gene_init", random.randint, 1, 100)  # n_slow
    toolbox.register("threshold_gene_init", random.gauss, 0, 25)
    toolbox.register("w100r_threshold_gene_init", uniform_random_scaled, -100)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
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

    with open("nasdaq.json", "r") as f:
        ticker_list = json.load(f)
    stocks = yf.Tickers(ticker_list)
    history = stocks.history(interval="1d", period="5y", start="2005-01-01")
    history = history.dropna(
        axis=1
    )  # discard stocks that have prices missing (maybe they were added later than 2014)
    history = history.swaplevel("Price", "Ticker", 1)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, history=history)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, -10.0))
    # final_pop = algorithm(toolbox)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    halloffame = tools.HallOfFame(25)
    initial_pop = toolbox.population(n=100)
    final_pop, logbook = algorithms.eaSimple(
        initial_pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=halloffame,
        verbose=True,
    )
    for ind in halloffame:
        print(ind)
        print(toolbox.evaluate(ind))
    print(logbook)
    plot(logbook)


if __name__ == "__main__":
    main()
