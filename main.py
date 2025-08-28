from pathlib import Path
from deap import base, tools, algorithms
import numpy as np
from individual import register_methods
from indicators import query_stock_exchange_history
import matplotlib.pyplot as plt
import uuid
import networkx
import random


def create_statistic_tool():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    return stats


def plot_pareto_front(pareto_halloffame, results_dir):
    Path("results").mkdir(exist_ok=True)
    for rank, ind in enumerate(pareto_halloffame):
        print(f"{rank + 1}")
        print(ind)
        print(ind.fitness.values)
        print("-------O-------")

    fitnesses = np.array([list(ind.fitness.values) for ind in pareto_halloffame])

    plt.figure(figsize=(8, 6))
    plt.scatter(fitnesses[:, 0], fitnesses[:, 1], c="red", s=80, label="Pareto front")
    plt.xlabel("Profit (maximize)")
    plt.ylabel("Number of Transactions (minimize)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/pareto.png")


def plot_results(logbook, results_dir):
    gen = logbook.select("gen")

    # stats multiobj
    fit_mins = logbook.select("min")  # [(f1_min, f2_min), ...]
    fit_avgs = logbook.select("avg")  # [(f1_avg, f2_avg), ...]
    fit_maxs = logbook.select("max")  # [(f1_max, f2_max), ...]

    # two objectives
    profit_min = [f[0] for f in fit_mins]
    transactions_min = [f[1] for f in fit_mins]

    profit_avg = [f[0] for f in fit_avgs]
    transactions_avg = [f[1] for f in fit_avgs]

    profit_max = [f[0] for f in fit_maxs]
    transactions_max = [f[1] for f in fit_maxs]

    plt.figure(figsize=(10, 5))
    plt.plot(gen, profit_min, "b--", label="Min profit")
    plt.plot(gen, profit_avg, "b-", label="Avg profit")
    plt.plot(gen, profit_max, "b:", label="Max profit")
    plt.xlabel("Generation")
    plt.ylabel("Profit")
    plt.title("Evolution of profit (maximizing)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/profit.png")

    plt.figure(figsize=(10, 5))
    plt.plot(gen, transactions_min, "r--", label="Min num transactions")
    plt.plot(gen, transactions_avg, "r-", label="Avg num transactions")
    plt.plot(gen, transactions_max, "r:", label="Max num transactions")
    plt.xlabel("Generation")
    plt.ylabel("Number of transactions")
    plt.title("Evolution of the number of transactions (minimizing)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/transactions.png")


def plot_generations(history, toolbox, results_dir):
    plt.close("all")
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()  # Make the graph top-down
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.savefig(f"{results_dir}/history.png")


def main():
    random.seed(1574147)
    toolbox = base.Toolbox()
    stock_history = query_stock_exchange_history()
    register_methods(toolbox, stock_history)
    stats = create_statistic_tool()

    pareto_halloffame = tools.ParetoFront()
    history = tools.History()

    initial_pop = toolbox.population(n=100)  # pyright: ignore[reportAttributeAccessIssue]
    history.update(initial_pop)

    toolbox.register("mate", tools.cxUniform)
    toolbox.register("select", tools.selNSGA2)

    # for history graph
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # _, logbook = algorithms.eaSimple(
    #     initial_pop,
    #     toolbox,
    #     cxpb=0.5,
    #     mutpb=0.2,
    #     ngen=100,
    #     stats=stats,
    #     halloffame=pareto_halloffame,
    #     verbose=True,
    # )

    # _, logbook = algorithms.eaMuPlusLambda(
    #     initial_pop,
    #     toolbox,
    #     mu=50,
    #     lambda_=100,
    #     cxpb=0.5,
    #     mutpb=0.2,
    #     ngen=100,
    #     stats=stats,
    #     halloffame=pareto_halloffame,
    #     verbose=True,
    # )

    _, logbook = algorithms.eaMuCommaLambda(
        initial_pop,
        toolbox,
        mu=50,
        lambda_=100,
        cxpb=0.5,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=pareto_halloffame,
        verbose=True,
    )

    results_dir = f"results/{uuid.uuid4()}"
    print(f"Saving results to {results_dir}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    plot_results(logbook, results_dir)
    plot_pareto_front(pareto_halloffame, results_dir)
    plot_generations(history, toolbox, results_dir)


if __name__ == "__main__":
    main()
