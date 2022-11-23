from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable
from typing import Generic
from typing import TypeVar
from typing import Union


FitnessType = Union[float, list[float]]
P = TypeVar("P")


SingleObjectiveCallable = Callable[[P], float]
MultiObjectiveCallable = Callable[[P], list[float]]


class SingleObjectiveWrapper(Generic[P]):
    def __init__(self, fun: Callable[[P], float]):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


class Problem(ABC):
    """An Abstract class that SingleObjectiveProblem and MultiObjectiveProblem
    extends to.

    Args:
        minimize (bool | list[bool]): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], float] | Callable[[P], list[float]]): The fitness function. Should take in any valid individual and return a float or a list of floats, depending if its a single objetive problem or a multi objective problem.
    """

    minimize: bool | list[bool]
    fitness_function: SingleObjectiveCallable | MultiObjectiveCallable

    def evaluate(self, phenotype: P) -> FitnessType:
        ...

    def solved(self, best_fitness: FitnessType):
        return False

    def overall_fitness(self, p: P) -> float:
        """Returns the evaluation of an individual, combining all scores.

        This function is going to be maximized.
        """
        ...


@dataclass
class SingleObjectiveProblem(Problem):
    """SingleObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (bool): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], float]): The fitness function. Should take in any valid individual and return a float.
        target_fitness (Optional[float]): Sets a target fitness. When this fitness is reached, the algorithm stops running (default = None).
    """

    minimize: bool
    fitness_function: Callable[[P], float]
    target_fitness: float | None = None

    def __init__(
        self,
        fitness_function: SingleObjectiveCallable,
        minimize: bool = False,
        target_fitness: float | None = None,
    ):
        self.fitness_function = fitness_function
        self.minimize = minimize
        self.target_fitness = target_fitness

    def evaluate(self, phenotype: P) -> float:
        c: SingleObjectiveCallable = self.fitness_function
        return float(c(phenotype))

    def overall_fitness(self, phenotype: P) -> float:
        if self.minimize:
            return -self.evaluate(phenotype)
        else:
            return self.evaluate(phenotype)

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
    """MultiObjectiveProblem is a class that extends the Problem class.

    Args:
        minimize (list[bool]): When switch on, the fitness function is reversed, so that a higher result from the fitness function corresponds to a less fit solution.
        fitness_function (Callable[[P], list[bool]]): The fitness function. Should take in any valid individual and return a list of float.
        best_individual_criteria_function (Optional(Callable[[P], float]): This function allow the user to choose how to find the best individual in a generation (default = None , this means that the individual with the best fitness is the one considered as the best in that generation)
    """

    minimize: list[bool]
    fitness_function: MultiObjectiveCallable
    best_individual_criteria_function: SingleObjectiveCallable | None = None

    def number_of_objectives(self):
        return len(self.minimize)

    def evaluate(self, phenotype: P) -> list[float]:
        return [float(x) for x in self.fitness_function(phenotype)]

    def overall_fitness(self, phenotype: P) -> float:
        if self.best_individual_criteria_function is None:
            return sum(m and -fit or +fit for (fit, m) in zip(self.evaluate(phenotype), self.minimize))
        else:
            return self.best_individual_criteria_function(phenotype)


def wrap_depth_minimization(p: SingleObjectiveProblem) -> SingleObjectiveProblem:
    """This wrapper takes a SingleObjectiveProblem and adds a penalty for
    bigger trees."""

    def w(i):
        if p.minimize:
            return p.fitness_function(i) + i.gengy_distance_to_term * 10**-25
        else:
            return p.fitness_function(i) - i.gengy_distance_to_term * 10**-25

    return SingleObjectiveProblem(
        minimize=p.minimize,
        fitness_function=w,
        target_fitness=None,
    )
