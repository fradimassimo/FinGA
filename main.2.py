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

stock = yf.Ticker("AAPL")
price = stock.history(interval="1d", period="1y", start="2012-01-01")["Close"]


def evaluate(individual, price):
    macd_short = individual[0]
    macd_long = individual[1]
    df = macd_crossover_indicator(macd_short, macd_long, 9, price)
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
    return (sum(profits),)


def gene_mutation(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.randint(5, 20)
    if random.random() < indpb:
        individual[1] = random.randint(21, 50)

    return (individual,)


toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", gene_mutation, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, price=price)


def macd_crossover_indicator(ema_short_days, ema_long_days, signal_days, price):
    """
    shows trends, if the short ema dips below the long ema the trend is going down
    if the short ema goes above the long, the trend is going up
    """
    ema_short = ema_indicator(ema_short_days, price)
    ema_long = ema_indicator(ema_long_days, price)

    macd = ema_short - ema_long

    signal_curve = ema_indicator(signal_days, macd)
    macd_crossover = macd - signal_curve

    macd_crossover = macd_crossover > 0
    return pd.DataFrame({"macd_crossover": macd_crossover, "price": price})


def ema_indicator(days, price):
    return price.ewm(span=days, adjust=False).mean()


if __name__ == "__main__":
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

    print(pop)
