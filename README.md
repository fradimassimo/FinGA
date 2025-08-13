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

## Goals
phenotype:
- List of tickers and their ratio in the portfolio
- At least 1 ticker in portfolio

fitness:
- how much that portfolio increased in a specific time period, or in multiple timeperiods

genotype:
- properties for a strategy to be decided
- portfolio wieghting strategies
