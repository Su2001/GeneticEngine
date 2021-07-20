from dataclasses import dataclass
import sys
from copy import deepcopy

from typing import Annotated, Any, Dict, TypeVar, Tuple, List

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.representations.base import Representation
from geneticengine.core.tree import Node
from geneticengine.core.utils import get_arguments, isTerminal
from geneticengine.exceptions import GeneticEngineError


@dataclass
class ProcessedGrammar(object):
    g: Grammar
    distanceToTerminal: Dict[Node, int]


def random_individual(
    r: RandomSource, pg: ProcessedGrammar, depth: int = 5, starting_symbol: Any = None
):
    g = pg.g
    if depth < 0:
        raise GeneticEngineError("Recursion Depth reached")

    if starting_symbol is None:
        starting_symbol = g.starting_symbol

    if starting_symbol is int:
        return r.randint(-(sys.maxsize - 1), sys.maxsize)
    elif hasattr(starting_symbol, "__origin__"):
        if starting_symbol.__origin__ is list:  # List
            size = r.randint(0, depth)
            return [
                random_individual(r, g, depth, starting_symbol.__args__[0])
                for _ in range(size)
            ]
    if hasattr(starting_symbol, "__metadata__"):
        metahandler = starting_symbol.__metadata__[0]
        recursive_generator = lambda: random_individual(
            r, g, depth, starting_symbol.__args__[0]
        )
        return metahandler.generate(r, recursive_generator)
    if starting_symbol not in g.productions:
        raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

    valid_productions = g.productions[starting_symbol]

    valid_productions = [vp for vp in valid_productions if pg.distanceToTerminal[vp] <= depth]
    if not valid_productions:
        raise GeneticEngineError(f"No productions for non-terminal {starting_symbol}")
    rule = r.choice(valid_productions)
    args = [random_individual(r, pg, depth - 1, at) for (a, at) in get_arguments(rule)]
    node = rule(*args)
    node.depth = max([1] + [n.depth for n in args if hasattr(n, "depth")])
    node.nodes = 1 + sum([n.nodes for n in args if hasattr(n, "nodes")])
    return node


def mutate_inner(r: RandomSource, g: Grammar, i: Node) -> Node:
    c = r.randint(0, i.nodes - 1)
    # print(f"#Nodes: {i.nodes}, choice: {c}")
    if c == 0:
        ty = i.__class__.__bases__[1]
        replacement = random_individual(r, g, i.depth + 1, ty)
        return replacement
    else:
        for field in i.__annotations__:
            child = getattr(i, field)
            if hasattr(child, "nodes"):
                count = getattr(i, field).nodes
                if c <= count:
                    setattr(i, field, mutate_inner(r, g, getattr(i, field)))
                    return i
                else:
                    c -= count
        return i


def mutate(r: RandomSource, g: Grammar, i: Node) -> Node:
    return mutate_inner(r, g, deepcopy(i))


def find_in_tree(ty: type, o: Node):
    if ty in o.__class__.__bases__:
        yield o
    for field in o.__annotations__:
        child = getattr(o, field)
        if hasattr(child, "__annotations__"):
            yield from find_in_tree(ty, child)


def tree_crossover_inner(
    r: RandomSource, g: Grammar, i: Node, o: Node
) -> Tuple[Node, Node]:
    c = r.randint(0, i.nodes - 1)
    if c == 0:
        ty = i.__class__.__bases__[1]
        replacement = r.choice(list(find_in_tree(ty, o)))
        if replacement is None:
            replacement = random_individual(r, g, i.depth + 1, ty)
        return (replacement, o)
    else:
        for field in i.__annotations__:
            child = getattr(i, field)
            if hasattr(child, "nodes"):
                count = getattr(i, field).nodes
                if c <= count:
                    setattr(i, field, tree_crossover_inner(r, g, getattr(i, field), o))
                    return (i, o)
                else:
                    c -= count
        return (i, o)

def tree_crossover(
    r: RandomSource, g: Grammar, p1: Node, p2: Node
) -> Tuple[Node, Node]:
    return tree_crossover_inner(r, g, deepcopy(p1), deepcopy(p2))

def preprocess_grammar(g: Grammar) -> ProcessedGrammar:
    choice = set()
    for k in g.productions.keys():
        choice.add(k)
    sequence = set()
    for vv in g.productions.values():
        for v in vv:
            if v not in choice:
                sequence.add(v)
    all_sym = sequence.union(choice)
    dist_to_terminal = {}
    for s in all_sym:
        dist_to_terminal[s] = 1000000
    changed = True
    while changed:
        changed = False
        for sym in all_sym:
            old_val = dist_to_terminal[sym]
            val = old_val
            if sym in choice:
                for prod in g.productions[sym]:
                    val = min(val, dist_to_terminal[prod])
            else:
                if hasattr(sym, "__annotations__"):
                    val = dist_to_terminal[sym.__annotations__.values().__iter__().__next__()]
                    for prod in sym.__annotations__.values():
                        val = max(val, dist_to_terminal[prod]+1)
                else:
                    val = 1
            if val != old_val:
                changed = True
                dist_to_terminal[sym] = val


    return ProcessedGrammar(g=g, distanceToTerminal=dist_to_terminal)

treebased_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=tree_crossover,
    preprocess_grammar=preprocess_grammar
)
