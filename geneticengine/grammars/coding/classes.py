from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar, Union

from geneticengine.core.decorators import abstract


class Statement(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda _: 0


t = TypeVar("t")


@abstract
class Expr(Generic[t]):
    def evaluate(self, **kwargs) -> t:
        ...

    def evaluate_lines(self, **kwargs) -> Callable[[Any], t]:
        ...


@abstract
class Number(Expr[float]):
    pass


@abstract
class Condition(Expr[bool]):
    pass


@abstract
class NumberList(Expr[List[float]]):
    pass


@dataclass
class XAssign(Statement):
    value: Expr

    def evaluate(self, **kwargs) -> float:
        if "x" not in kwargs:
            kwargs["x"] = 1.0
        return self.value.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: self.value.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return "x = {}".format(self.value)
