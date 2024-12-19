from __future__ import annotations

import logging

from geneticengine.algorithms.gp.population import Population
from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.algorithms.hybrid.initializer import LLMsInitializer
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import (
    ProgressTracker,
)
from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import (
    StandardInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithCrossover, RepresentationWithMutation, Representation


logger = logging.getLogger(__name__)


def default_generic_programming_step():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            NoveltyStep(),
            SequenceStep(
                TournamentSelection(5),
                GenericCrossoverStep(0.01),
                GenericMutationStep(0.9),
            ),
        ],
        weights=[5, 5, 90],
    )


class HybridSearch(HeuristicSearch):

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        tracker: ProgressTracker | None = None,
        population_size: int = 100,
        population_initializer: PopulationInitializer = None,
        step: GeneticStep | None = None,
    ):
        super().__init__(problem, budget, representation, random, tracker)
        self.population_size = population_size
        self.population_initializer = (
            population_initializer if population_initializer is not None else StandardInitializer()
        )
        self.llm_initializer = LLMsInitializer()
        self.step = step if step is not None else default_generic_programming_step()

    def perform_search(self) -> list[Individual] | None:
        assert isinstance(self.representation, RepresentationWithMutation)
        assert isinstance(self.representation, RepresentationWithCrossover)
        generation = 0
        logger.debug("Generating initial population")
        population = Population(
            self.llm_initializer.initialize(
                self.problem,
                self.representation,
                self.random,
                self.population_size,
            ),
            self.tracker,
            generation=generation,
        )

        while not self.is_done():
            generation += 1
            logger.debug(f"Generating population at generation {generation}")
            population = Population(
                self.step.apply(
                    self.problem,
                    self.tracker.evaluator,
                    self.representation,
                    self.random,
                    population.get_individuals(),
                    self.population_size,
                    generation,
                ),
                self.tracker,
                generation,
            )

        return self.tracker.get_best_individuals()
