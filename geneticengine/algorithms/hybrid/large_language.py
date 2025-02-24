from typing import Any
from ollama import generate
from geneticengine.grammar.utils import get_arguments, get_generic_parameter, get_generic_parameters, is_generic_list, is_generic_tuple, is_metahandler
import inspect
import re

def extract_content(text):
    text_no_white =re.sub(r'\s+', '', text)
    stack = []
    result = []
    end = 0
    start = 0

    while end < len(text_no_white):
        if text_no_white[end] in ['(']:
            stack.append(')')
        elif text_no_white[end] in ['[']:
            stack.append(']')
        elif len(stack) and text_no_white[end] == stack[-1]:
            stack.pop()
        elif not len(stack) and text_no_white[end] == ',':
            result.append(text_no_white[start:end])
            start = end+1
        end += 1
    result.append(text_no_white[start:end+1])   

    return result

def isBuiltin(ty: type):
    if ty in [int, bool, float, str, tuple, list, type] or is_generic_tuple(ty) or is_generic_list(ty):
        return True
    return False

def build(starting_symbol:type[Any], considered_subtypes: list[type], s:str):
    if starting_symbol is int:
        return int(s)
    elif starting_symbol is float:
        return float(s)
    elif starting_symbol is bool:
        return bool(s)
    elif starting_symbol is str:
        return eval(s)
    elif is_generic_tuple(starting_symbol):
        types = get_generic_parameters(starting_symbol)
        tup = extract_content(s[1:-1])
        vals = (build(t, considered_subtypes, inst) for inst, t in zip(tup,types))
        return vals
    elif is_generic_list(starting_symbol):
        lis = extract_content(s[1:-1])
        arr = [build(get_generic_parameter(starting_symbol), considered_subtypes, elem) for elem in lis]
        return arr
    elif is_metahandler(starting_symbol):
        return build(get_generic_parameter(starting_symbol), considered_subtypes, s)
    else:
        arr = []
        apply_symbol: type[Any] = starting_symbol
        for subtypes in considered_subtypes:
            if s.split("(",1)[0] == subtypes.__name__:
                apply_symbol = subtypes
                break
        contents = extract_content(re.findall(r"\((.*)\)", s)[0])
        for i, j in zip(contents, get_arguments(apply_symbol)):
            arr.append(build(j[1], considered_subtypes, i.split("=",1)[1]))

    return apply_symbol(*arr)

def getClassInformation(considered_subtypes: list[type]):
    classes = ""
    added = set()
    for argt in considered_subtypes:
        annotations = inspect.get_annotations(argt,eval_str=True)
        if argt not in added and not is_metahandler(argt) and not isBuiltin(argt):
            source = inspect.getsource(argt).split("\n")
            classes += f"{source[0]}\n{source[1]}\n"
            for name, annotation in annotations.items():
                classes += f"    {name}: {annotation}\n"
            classes += "\n"
            added.add(argt)
    return classes

def create(starting_symbol: type[Any], considered_subtypes: list[type], target_size: int):
    classes = getClassInformation(considered_subtypes)
    extra_requirements = "show all of them,one per line, with diversity, do not explain the result,do not enumerate it and do not generate python code.\n"
    asking = f"Given the following Python code, that uses dataclasses and custom annotations, please create exaclty 10 instantiations of class Level, {extra_requirements}\n\n```\n"
    count = 0
    while count < target_size:
        try:
            response = generate(
                model="qwen2.5:32b",
                prompt=f"{asking}{classes}```",
            )
            result = response.response.split("\n")
            for inst in result:
                yield build(starting_symbol, considered_subtypes, inst)
                count +=1
        except Exception:
            #ignoring error 
            continue

