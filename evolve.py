import random
import multiprocessing

from deap import base, creator, tools, algorithms
import numpy as np

import fitness

N_GEN = 40
POP_SIZE = 10
LAMBDA = 20  # offsprings
CXPB = 0.6  # crossover probability per pair
MUTPB = 0.3  # mutation probability per individual

TOURN_SIZE = 3
MUT_SIGMA = 0.2
MUT_INDPB = 0.15
BLEND_ALPHA = 0.3


def build_toolbox() -> base.Toolbox:
    # checks to ensure idempotency
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # weight initialization (small init prevents tanh saturation)
    toolbox.register("attr_float", random.gauss, 0.0, 0.5)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=fitness.genome_len,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # tournament selection
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

    # blend crossover
    toolbox.register("mate", tools.cxBlend, alpha=BLEND_ALPHA)

    # mutation: Gaussian perturbation
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0.0,
        sigma=MUT_SIGMA,  # step size
        indpb=MUT_INDPB,  # mutation probability per gene
    )

    toolbox.register("evaluate", fitness.evaluate)
    return toolbox


def evolve():
    random.seed(42)
    np.random.seed(42)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    toolbox = build_toolbox()
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=fitness.make_env) as pool:
        toolbox.register("map", pool.map)
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, POP_SIZE, LAMBDA, CXPB,
                                             MUTPB, N_GEN, stats, hof, verbose=True)


if __name__ == "__main__":
    evolve()
