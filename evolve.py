from deap import base, creator, tools, algorithms
from time import gmtime, strftime
from pathlib import Path
import multiprocessing
import numpy as np
import random
import pickle
import time

import fitness

POP_SIZE = 20
LAMBDA = 80  # offsprings
CXPB = 0.7  # crossover probability per pair
MUTPB = 0.3  # mutation probability per individual

TOURN_SIZE = 3
MUT_SIGMA = 0.2
MUT_INDPB = 0.15
BLEND_ALPHA = 0.4


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


SAVE_PERIOD = 5


def evolve(
        n_gen: int
):
    time_ = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    SAVE_DIR = Path(f"checkpoints/{time_}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
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

    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count(), initializer=fitness.make_env
    ) as pool:
        toolbox.register("map", pool.map)
        for i in range(n_gen // SAVE_PERIOD):
            start_time = time.time()
            pop, log = algorithms.eaMuPlusLambda(
                pop,
                toolbox,
                POP_SIZE,
                LAMBDA,
                CXPB,
                MUTPB,
                SAVE_PERIOD,
                stats,
                hof,
                verbose=True,
            )
            elapsed = time.time() - start_time
            print(f"Elapsed time: {elapsed}")

            with open(SAVE_DIR / "best-ever.pkl", "wb") as f:
                pickle.dump(hof[0], f)

            with open(SAVE_DIR / f"epoch-pop-{i}.pkl", "wb") as f:
                pickle.dump(pop, f)

            with open(SAVE_DIR / f"epoch-log-{i}.pkl", "wb") as f:
                pickle.dump(log, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evolve script")
    parser.add_argument(
        "--tries", default=1, type=int, help="Number of tests per agent in fitness"
    )
    parser.add_argument("n_gen", type=int)
    args = parser.parse_args()

    fitness.tries = args.tries
    print(f"Running: gens={args.n_gen}, tries={fitness.tries}")

    evolve(args.n_gen)
