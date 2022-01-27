from typing import (
    Any,
    Callable,
    Dict,
    Type,
)

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar


class VarRange(MetaHandlerGenerator):
    def __init__(self, options):
        self.options = options

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        newsymbol,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        rec(r.choice(self.options))

    def __repr__(self):
        return str(self.options)
