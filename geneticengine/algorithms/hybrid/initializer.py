from __future__ import annotations
from typing import Iterator, TypeVar

from geneticengine.algorithms.hybrid.large_language import create
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation

G = TypeVar("G")

class LLMsInitializer(PopulationInitializer):
    """All individuals are created with full trees (maximum depth in all
    branches)."""

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        for i in create(representation.__getattribute__("grammar").__getattribute__("starting_symbol"), target_size):
            yield PhenotypicIndividual(
                i,
                representation=representation,
            )
