from typing import Any
from litellm import completion
from geneticengine.grammar.utils import get_arguments, get_generic_parameter, get_generic_parameters, is_generic_list, is_generic_tuple, is_metahandler
import inspect
import re

## set ENV variables

def isBuiltin(ty: type):
    if ty in [int, bool, float, str, tuple, list, type] or is_generic_tuple(ty) or is_generic_list(ty):
        return True
    return False

def build(starting_symbol:type[Any], s:str):
    if starting_symbol is int:
        return int(s)
    elif starting_symbol is float:
        return float(s)
    elif starting_symbol is bool:
        return bool(s)
    elif is_generic_tuple(starting_symbol):
        types = get_generic_parameters(starting_symbol)
        tup = re.split(r', |\(|\)', s)
        vals = (build(t, inst) for inst, t in zip(tup[1:-1],types))
        return vals
    elif is_generic_list(starting_symbol):
        arr = []
        lis = re.split(r', |\[|\]', s)
        for elem in lis[1:-1]:
            arr.append(build(get_generic_parameter(starting_symbol), elem))
        return arr
    elif is_metahandler(starting_symbol):
        return build(get_generic_parameter(starting_symbol),s)
    else:
        arr = []
        a:str = re.findall(r"\((.*)\)", s)[0]
        b = re.split(r', (?=(?:[^()\[\]]*(?:\([^()\[\]]*\)|\[[^()\[\]]*\]))*[^()\[\]]*$)', a)
        for i, j in zip(b, get_arguments(starting_symbol)):
            arr.append(build(j[1], i))

    return starting_symbol(*arr)

def create(starting_symbol: type[Any], targetsize: int):
    classes = ""
    extra_requirements = "show all of them,one per line, dont show keyword argument and dont need be assigned only the instance, with diversity and do not generate python code."
    asking = f"Given the following Python code, that uses dataclasses and custom annotations, please create exaclty {targetsize} instantiations of class {starting_symbol}, {extra_requirements}\n\n```\n"
    added = set()
    for _, argt in get_arguments(starting_symbol):
        if argt not in added and not is_metahandler(argt) and not isBuiltin(argt):
            classes += inspect.getsource(argt)
            added.add(argt)
    classes += inspect.getsource(starting_symbol)
    print(f"{asking}{classes}``")
    response = completion(
        model="ollama/llama3.1",
        messages=[{ "content": f"{asking}{classes}``","role": "user"}],
        api_base="http://localhost:11434",
    )
    p =response.choices[0].message.content
    print(p)
    result =re.split(r'```', p)[1].split("\n")[1:-1]
    print(result)
    for inst in result:
        yield build(starting_symbol, inst)

#for test
extra_requirements = "show all of them,one per line, do not response this, I only want results, with diversity, do not enumerate it and do not generate python code.\n"
asking = f"Given the following Python code, that uses dataclasses and custom annotations, please create exaclty 10 instantiations of class A, {extra_requirements}\n\n```\n"
last = "you should generate in this format A(x,y,z,q,p)\nYou must obey the rule of annotation.\nThe annotation IntRange means that x: Annotated[int, IntRange(a,b)] restricts x's value must to be between a and b. So IntRange(30,50) equals 30<= x<=50\nThe annotation IntRange means that x: Annotated[list[int], ListSizeBetween(a,b)] you need generate  a list of int with size in a to b"
classes = """
@dataclass
class A():
    nums: Annotated[int, IntRange(350,800)]
    num2: Annotated[int, IntRange(30,50)]
    num3: Annotated[int, IntRange(136,190)]
    num4: Annotated[list[int], ListSizeBetween(2,6)]
    num5: Annotated[int, IntRange(-100,0)]"""
print(f"{asking}{classes}``{last}")
with open("output2.txt", "w") as file:
    for i in range(10):
        response = completion(
                model="ollama/codellama",
                messages=[{ "content": f"{asking}{classes}```\n{last}","role": "user"}],
                api_base="http://localhost:11434",
        )
        p =response.choices[0].message.content
        print(p, file=file)
        print(f"done{i}")
print("done")
