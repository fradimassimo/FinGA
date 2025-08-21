# FinGA
Implementation of a genetic algorithm (GA) for the optimization of financial portfolio allocation.

## To run the program
```bash
# use python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# running the algorithm
python main.py
```

## TODO
- add logging of fitness of initial and final population
- add graphs
- punish invalid individual fitness
- adding more stocks, more timeperiods, getting a combined profit
- genetic programming ()

## Goals
Single stock, the goal is to generate a buy signal and a sell signal
buying has to be first, we cannot buy if we already bought before, selling has to be after the buy, when we sold, we calculate fitness

phenotype:
- buy signal
- sell signal

fitness:
- (sell_price - buy_cost) / days

Genotype:
- parameters of the indicator
- Thresholds for the buy and sell signal


Future:
- combine multiple indicators
- combine with genetic programming
- Fitness should be an average of multiple scenarios
