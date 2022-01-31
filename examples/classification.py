import os
from typing import Annotated, Any, Callable

import numpy as np
import pandas as pd
from math import isinf

from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, Var
from geneticengine.grammars.basic_math import SafeLog, SafeSqrt, Sin, Tanh, Exp, SafeDiv
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import f1_score

DATASET_NAME = "Banknote"
DATA_FILE_TRAIN = "examples/data/{}/Train.csv".format(DATASET_NAME)
DATA_FILE_TEST = "examples/data/{}/Test.csv".format(DATASET_NAME)

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
# import IPython as ip
# ip.embed()
target = bunch.y
data = bunch.drop(["y"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


def preprocess():
    return extract_grammar(
        [Plus, Mul, SafeDiv, Literal, Var, Exp, SafeSqrt, Sin, Tanh, SafeLog], Number
    )


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]
    y_pred = n.evaluate(**variables)

    if type(y_pred) in [np.float64, int, float]:
        """If n does not use variables, the output will be scalar."""
        y_pred = np.full(len(y), y_pred)
    if y_pred.shape != (len(y),):
        return -100000000
    fitness = f1_score(y_pred, y)
    if isinf(fitness):
        fitness = -100000000
    return fitness


def evolve(g, seed, mode, representation='treebased_representation', output_folder=('','all')):
    if representation == 'grammatical_evolution':
        representation = ge_representation
    else:
        representation = treebased_representation
    
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        minimize=False,
        selection_method=("tournament", 2),
        max_depth=17,
        population_size=500,
        # max_init_depth=10,
        # mutation uses src.operators.mutation.int_flip_per_ind. As mutation prob is None, the probability becomes 1/genome_length per codon (what is this?). How do we translate that to our method?
        number_of_generations=50,
        probability_crossover=0.75,
        n_elites=5,
        seed=seed,
        timer_stop_criteria=mode,
        target_fitness=1,
        safe_gen_to_csv=output_folder
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
