from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pickle import bytes_types
from tokenize import Single
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union


FitnessType = Union[float, list[float]]
P = TypeVar("P")


class Problem(ABC):
    def evaluate(self, p: P) -> FitnessType:
        ...

    def solved(self, best_fitness: FitnessType):
        return False


@dataclass
class SingleObjectiveProblem(Problem):
    minimize: bool
    fitness_function: Callable[[P], float]
    target_fitness: float | None

    def evaluate(self, p: P) -> float:
        return float(self.fitness_function(p))

    def solved(self, best_fitness: FitnessType):
        assert isinstance(best_fitness, float)
        if not self.target_fitness:
            return False
        elif self.minimize:
            return best_fitness <= self.target_fitness
        else:
            return best_fitness >= self.target_fitness


@dataclass
class MultiObjectiveProblem(Problem):
    minimize: list[bool]
    fitness_function: Callable[[P], list[float]]

    def number_of_objectives(self):
        return len(self.minimize)

    def evaluate(self, p: P) -> list[float]:
        return [float(x) for x in self.fitness_function(p)]


def wrap_depth_minimization(p: SingleObjectiveProblem) -> SingleObjectiveProblem:
    """
    This wrapper takes a SingleObjectiveProblem and adds a penalty for bigger trees.
    """

    def w(i):
        if p.minimize:
            return p.fitness_function(i) + i.genotype.gengy_distance_to_term * 10**-25
        else:
            return p.fitness_function(i) - i.genotype.gengy_distance_to_term * 10**-25

    return SingleObjectiveProblem(
        minimize=p.minimize,
        fitness_function=w,
        target_fitness=None,
    )


def process_problem(
    problem: Problem | None,
    evaluation_function: Callable[[P], float] = None,  # DEPRECATE in the next version
    minimize: bool = False,  # DEPRECATE in the next version
    target_fitness: float | None = None,  # DEPRECATE in the next version
) -> Problem:
    """
    This function is a placeholder until we deprecate all the old usage of GP class.
    """
    if problem:
        return problem
    elif isinstance(minimize, list) and evaluation_function:
        return MultiObjectiveProblem(minimize, evaluation_function)
    elif isinstance(minimize, bool) and evaluation_function:
        return SingleObjectiveProblem(minimize, evaluation_function, target_fitness)
    else:
        raise NotImplementedError(
            "This combination of parameters to define the problem is not valid",
        )


def wrap_depth(p: Problem, favor_less_deep_trees: bool = False):
    if isinstance(p, SingleObjectiveProblem):
        if favor_less_deep_trees:
            return wrap_depth_minimization(p)
        else:
            return p
    else:
        assert isinstance(p, MultiObjectiveProblem)
        return p
