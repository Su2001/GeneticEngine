from abc import ABC, abstractmethod
from typing import Optional
from geneticengine.evaluation.exceptions import IndividualFoundException
from geneticengine.evaluation.budget import SearchBudget

from geneticengine.evaluation.tracker import (
    ProgressTracker,
)
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import Individual


class SynthesisAlgorithm(ABC):
    tracker: ProgressTracker
    problem: Problem
    budget: SearchBudget
    representation: Representation

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        tracker: Optional[ProgressTracker] = None,
    ):
        self.problem = problem
        self.budget = budget
        self.representation = representation

        if tracker is None:
            if isinstance(problem, SingleObjectiveProblem):
                self.tracker = ProgressTracker(problem, SequentialEvaluator())
            else:
                self.tracker = ProgressTracker(problem, SequentialEvaluator())
        else:
            self.tracker = tracker

        assert self.tracker is not None

    def is_done(self) -> bool:
        """Whether the synthesis should stop, or not."""
        return self.budget.is_done(self.tracker)

    def get_problem(self) -> Problem:
        return self.problem

    def search(self) -> list[Individual] | None:
        try:
            return self.perform_search()
        except IndividualFoundException as e:
            return [e.individual]

    @abstractmethod
    def perform_search(self) -> list[Individual] | None: ...
