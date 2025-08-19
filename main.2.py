from deap import base, creator, tools
import random

#PROBLEM DEFINITION
'''
Creator module allows to define/build your own problem
'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Minimization problem -> weights = (-1.0)
creator.create("Individual", list, fitness=creator.FitnessMax)

#INITIALIZING POP
'''
Using tools module, population is initialized following this hierarchy:
 (gene)->(individuals)->(pop) 
'''
IND_SIZE = 1 #NB using initRepeat = number of genes for each individual [treshold, pI1, pI2] but in the case below is number of cycles
toolbox = base.Toolbox()

#MACD encoding
'''
MACD is built using these ingredients:
fast EMA , slow EMA, signal line 
MACD = EMA_short − EMA_long
Signal = EMA(MACD, 9)
'''

toolbox.register("gene1", random.randint, 5, 20) # n_fast
toolbox.register("gene2", random.randint, 21, 50) # n_slow


toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.gene1, toolbox.gene2),
    n=IND_SIZE
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def MACD_indicator(EMA_short, EMA_long, ticker, current_day):





def evaluate(individual):
    return sum(individual),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    pop = toolbox.population(n=5)
    for ind in pop:
        print(ind)