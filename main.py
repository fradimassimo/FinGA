import json
import numpy as np
import yfinance as yf
import pandas as pd

from leap_ec import Individual, Representation
from leap_ec.algorithm import generational_ea, stop_at_generation
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
import leap_ec.ops as ops
from leap_ec.problem import ScalarProblem

pop_size = 100


def solve(pop_size, max_generations, price_diff):
    final_pop = generational_ea(
        max_generations=max_generations,
        pop_size=pop_size,
        problem=Portfolio(price_diff),
        representation=Representation(
            initialize=create_binary_sequence(
                length=len(price_diff)
            )  # Initial genomes are random binary sequences
        ),
        # The operator pipeline
        pipeline=[
            ops.tournament_selection,  # Select parents via tournament selection
            ops.clone,  # Copy them (just to be safe)
            mutate_bitflip(
                expected_num_mutations=1
            ),  # Basic mutation with a 1/L mutation rate
            ops.UniformCrossover(
                p_swap=0.4
            ),  # Crossover with a 40% chance of swapping each gene
            ops.evaluate,  # Evaluate fitness
            ops.pool(size=pop_size),  # Collect offspring into a new population
        ],
    )
    return final_pop


class Portfolio(ScalarProblem):
    def __init__(self, price_diff: pd.DataFrame):
        super().__init__(maximize=True)
        self.price_diff = price_diff

    def evaluate(self, portfolio) -> float:
        return self.price_diff[np.array(portfolio, dtype=bool)].mean().item()


def decode_pop(final_pops, price_diff):
    result = [
        (pop.fitness, price_diff.index[pop.phenome].tolist()) for pop in final_pops
    ]
    return sorted(result, key=lambda x: x[0])


def get_nasdaq_tickers():
    with open("nasdaq.json", "r") as f:
        tickers = json.load(f)

    return tickers


def get_prices(tickers):
    df_prices = yf.Tickers(tickers)
    history = df_prices.history()
    return history


def get_price_increase(history):
    start_price = history["Close"].iloc[0]
    end_price = history["Close"].iloc[-1]
    return 100 * (end_price - start_price) / start_price


if __name__ == "__main__":
    np.random.seed(100)
    tickers = get_nasdaq_tickers()
    history = get_prices(tickers)
    price_diff = get_price_increase(history)
    final_pops = solve(100, 100, price_diff)
    print(decode_pop(final_pops, price_diff))
