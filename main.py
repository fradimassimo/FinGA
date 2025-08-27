from deap import base, tools, algorithms
import numpy as np
from individual import register_methods
from indicators import query_stock_exchange_history


def create_statistic_tool():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    return stats


def main():
    toolbox = base.Toolbox()
    stock_history = query_stock_exchange_history()
    register_methods(toolbox, stock_history)
    stats = create_statistic_tool()

    halloffame = tools.HallOfFame(25)

    initial_pop = toolbox.population(n=100)  # pyright: ignore[reportAttributeAccessIssue]

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    algorithms.eaSimple(
        initial_pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=halloffame,
        verbose=True,
    )
    for rank, ind in enumerate(halloffame):
        print(f"----------------{rank + 1}---------------")
        print(ind)
        print(ind.fitness.values)


if __name__ == "__main__":
    main()
