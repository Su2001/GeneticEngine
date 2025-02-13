from __future__ import annotations

import logging

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.hybrid.initializer import LLMsInitializer
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import (
    ProgressTracker,
)
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation


logger = logging.getLogger(__name__)



class HybridSearch(GeneticProgramming):

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        tracker: ProgressTracker | None = None,
        population_size: int = 100,
        step: GeneticStep | None = None,
    ):
        super().__init__(problem, budget, representation, random, tracker, population_size, LLMsInitializer(), step)
