from typing import Annotated, Any, Callable
from examples.progsys.utils import get_data, import_embedded

from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.coding.classes import Number, Statement, XAssign
from geneticengine.grammars.coding.numbers import Max, Min, Abs, Plus, Literal, Mul, SafeDiv, Var
from geneticengine.grammars.coding.control_flow import IfThen, IfThenElse, While
from geneticengine.grammars.coding.conditions import Equals, NotEquals, GreaterOrEqualThan, GreaterThan, LessOrEqualThan, LessThan, Is, IsNot
from geneticengine.grammars.coding.logical_ops import And, Or
from geneticengine.metahandlers.vars import VarRange
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation

FILE_NAME = "Smallest"
DATA_FILE_TRAIN = "./examples/progsys/data/{}/Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "./examples/progsys/data/{}/Test.txt".format(FILE_NAME)

inval, outval = get_data(DATA_FILE_TRAIN, DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

vars = ["in0", "in1", "in2", "in3"]
variables = {}
for i, n in enumerate(vars):
    variables[n] = i

XAssign.__annotations__["value"] = Number
Var.__annotations__["name"] = Annotated[str, VarRange(vars)]
Var.feature_indices = variables
g = extract_grammar(
    [
        Plus,
        Literal,
        Mul,
        SafeDiv,
        Max,
        Min,
        Abs,
        And,
        Or,
        Var,
        Equals,
        NotEquals,
        GreaterOrEqualThan,
        GreaterThan,
        LessOrEqualThan,
        LessThan,
        Is,
        IsNot,
        XAssign,
        IfThen,
        IfThenElse  #, While
    ],
    Statement)
print("Grammar: {}.".format(repr(g)))


def fitness_function(n: Statement):
    fitness, error, cases = imported.fitness(inval, outval, n.evaluate_lines())
    return fitness


def preprocess():
    return extract_grammar(
        [
            Plus,
            Literal,
            Mul,
            SafeDiv,
            Max,
            Min,
            Abs,
            And,
            Or,
            Var,
            Equals,
            NotEquals,
            GreaterOrEqualThan,
            GreaterThan,
            LessOrEqualThan,
            LessThan,
            Is,
            IsNot,
            XAssign,
            IfThen,
            IfThenElse  #, While
        ],
        Statement)


def evolve(g, seed, mode):
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        number_of_generations=50,
        minimize=True,
        seed=seed,
        max_depth=17,
        population_size=500,
        probability_crossover=0.9,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf


if __name__ == '__main__':
    g = preprocess()
    print("Grammar: {}.".format(repr(g)))
    b, bf = evolve(g, 0)
    print(b, bf)