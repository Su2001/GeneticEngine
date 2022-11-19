from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Annotated
from typing import List
from typing import Tuple

from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.metahandlers.lists import ListSizeBetween

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# The Santa Fe Trail problem is a GP problem in which artificial ants search for food pellets according to a programmed set of instructions
# ===================================

map = """.###............................
...#............................
...#.....................###....
...#....................#....#..
...#....................#....#..
...####.#####........##.........
............#................#..
............#.......#...........
............#.......#........#..
............#.......#...........
....................#...........
............#................#..
............#...................
............#.......#.....###...
............#.......#..#........
.................#..............
................................
............#...........#.......
............#...#..........#....
............#...#...............
............#...#...............
............#...#.........#.....
............#..........#........
............#...................
...##. .#####....#...............
.#..............#...............
.#..............#...............
.#......#######.................
.#.....#........................
.......#........................
..####..........................
................................"""


class ActionMain(ABC):
    pass


class Action(ABC):
    pass


@dataclass
class ActionBlock(ActionMain):
    actions: Annotated[list[Action], ListSizeBetween(2, 30)]


@dataclass
class IfFood(Action):
    yes_food: Action
    no_food: Action


@dataclass
class Move(Action):
    pass


@dataclass
class Right(Action):
    pass


@dataclass
class Left(Action):
    pass


class Direction(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class Position(Enum):
    EMPTY = 0
    FOOD = 1


def map_from_string(map_str: str) -> list[list[Position]]:
    return [
        [pos == "#" and Position.FOOD or Position.EMPTY for pos in line]
        for line in map_str.split("\n")
    ]


def next_pos(
    pos: tuple[int, int, Direction],
    map: list[list[Position]],
) -> tuple[int, int]:
    masks = {
        Direction.EAST: (0, 1),
        Direction.SOUTH: (1, 0),
        Direction.WEST: (0, -1),
        Direction.NORTH: (-1, 0),
    }
    row = (pos[0] + masks[pos[2]][0]) % len(map)
    col = (pos[1] + masks[pos[2]][1]) % len(map[0])
    return (row, col)


def food_in_front(pos: tuple[int, int, Direction], map: list[list[Position]]) -> bool:
    (row, col) = next_pos(pos, map)
    return map[row][col] == Position.FOOD


def simulate(a: Action, map_str: str) -> int:
    next_instructions: list[Action] = [a]
    food_consumed = 0
    map = map_from_string(map_str)
    current_pos: tuple[int, int, Direction] = (
        0,
        0,
        Direction.EAST,
    )  # row, col, direction
    while next_instructions:
        current_instruction = next_instructions.pop(0)  # Default is -1
        if isinstance(
            current_instruction,
            ActionBlock,
        ):  # ActionBlock contains list of action lists.
            for action in reversed(current_instruction.actions):
                next_instructions = [action] + next_instructions
        elif isinstance(current_instruction, IfFood):
            if food_in_front(current_pos, map):
                next_instructions.insert(0, current_instruction.yes_food)
            else:
                next_instructions.insert(0, current_instruction.no_food)
        elif isinstance(current_instruction, Move):
            (row, col) = next_pos(current_pos, map)
            if map[row][col] == Position.FOOD:
                food_consumed += 1
                map[row][col] = Position.EMPTY
            current_pos = (row, col, current_pos[2])
        elif isinstance(current_instruction, Left):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value - 1) % 4),
            )
        elif isinstance(current_instruction, Right):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value + 1) % 4),
            )
    return food_consumed


def preprocess():
    return extract_grammar([ActionBlock, Action, IfFood, Move, Right, Left], ActionMain)


def evolve(
    fitness_function,
    g,
    seed,
    mode,
    representation="TreeBasedRepresentation",
):
    if representation == "ge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "sge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "dsge":
        representation = GrammaticalEvolutionRepresentation
    else:
        representation = TreeBasedRepresentation

    alg = SimpleGP(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        # number_of_generations=150,
        # population_size=100,
        # max_depth=15,
        # favor_less_deep_trees=True,
        # probability_crossover=0.75,
        # probability_mutation=0.01,
        # selection_method=("tournament", 2),
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


def fitness_function(i) -> float:
    return simulate(i, map)


if __name__ == "__main__":
    g = preprocess()
    print(f"Grammar: {repr(g)}")
    (b_gp, bf_gp) = evolve(lambda p: simulate(p, map), g, 123, False, "dsge")

    alg_hc = HC(
        representation=TreeBasedRepresentation(g, 10),
        problem=SingleObjectiveProblem(fitness_function),
        stopping_criterium=GenerationStoppingCriterium(50),
    )
    (b_hc, bf_hc, bp_hc) = alg_hc.evolve()

    print("\n======\nHC\n======\n")
    print(bf_hc, bp_hc, b_hc)

    print("\n======\nGP\n======\n")
    print(bf_gp, b_gp)
