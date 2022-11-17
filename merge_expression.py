#!/usr/bin/env python3

import sys
from dataclasses import dataclass

import sys

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Tuple, Type, Dict
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar


class Tens():
    pass


@dataclass()
class Var(Tens):
    name: str


@dataclass()
class Factored():
    item: Tens
    factor: Optional[Union[int, float]] = None


@dataclass()
class Merge(Tens):
    items: List[Factored]


@dataclass()
class Minus(Tens):
    left: Tens
    right: Tens


def ensure(truthy, msg = None, *args):
    if not truthy:
        raise ValueError("ensure truthy error", *([msg] or []), truthy, *args)


def ensure_equal(val, expected, msg = None, *args):
    if val != expected:
        raise ValueError("ensure equal error", *([msg] or []), val, expected, *args)


def ensure_type(item, expected: Union[Type, Iterable[Type]], msg = None, *args):
    try:
        expected = list(expected)
    except:
        expected = [expected]

    for t in expected:
        if isinstance(item, t):
            return

    raise ValueError("ensure type error", *([msg] or []), type(item), expected, item, *args)


def var_parse_action(section, pos, res):
    print("var_parse_action", (pos, res))
    ensure_equal(len(res), 1)
    ensure_type(res[0], str)
    return Var(res[0])


def mul_parse_action(section, pos, full):
    # print("mul_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]
    # for pos, i in enumerate(res):
    #     print(pos, i)
    ensure_equal(len(res), 3, "mul only allowed once for variable/expression", len(res), pos, res)
    ensure_type(res[0], Tens, "left mul op must be Tensor", res[0], full)
    ensure_equal(res[1], "*")
    ensure_type(res[2], (int, float), "mul factor must be int/float")
    return Factored(res[0], res[2])


def minus_parse_action(section, pos, full):
    # print("minus_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]

    for pos, i in enumerate(res):
        if i == "-":
            continue
        # print(pos, i)
        ensure_type(i, Tens, "minus argument must be tensor", pos, type(i), i)

    # ensure(len(res) == 3, "minus only allowed once for variable/expression", len(res), pos, res)
    return Minus(res[0], res[2])


def plus_parse_action(section, pos, full):
    # print("plus_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]
    use = []
    for pos, i in enumerate(res):
        if i == "+":
            continue
        if isinstance(i, Merge):
            for a in i.items:
                ensure_type(a, Factored, "plus expanded arg must have factor", pos, type(i), i, full)
            use.extend(i.items)
        else:
            if isinstance(i, Tens):
                i = Factored(i, 1)
            ensure_type(i, Factored, "plus arg must have factor", pos, type(i), i, full)
            use.append(i)
    return Merge(use)


def infix_parse_action(section, pos, full):
    # print("infix_parse_action", (pos, full))
    # ensure_equal(len(full),1)
    # return Group(full[0])
    return full


def make_parser(enablePackrat: Optional[bool] = None):
    import pyparsing as pyp
    from pyparsing import ParserElement, pyparsing_common, Word

    if enablePackrat:
        ParserElement.enablePackrat()

    integer = pyparsing_common.integer
    variable = Word(pyp.alphas, pyp.alphanums + "._").set_parse_action(var_parse_action)
    operand = integer | variable

    multop = pyp.oneOf("*")
    plusop = pyp.oneOf("+")
    minusop = pyp.oneOf("-")

    # minusop = pyp.oneOf("-")
    # plusop = pyp.oneOf("+")

    expr = pyp.infix_notation(
        operand,
        [
            (multop, 2, pyp.opAssoc.LEFT, mul_parse_action),
            (minusop, 2, pyp.opAssoc.LEFT, minus_parse_action),
            (plusop, 2, pyp.opAssoc.LEFT, plus_parse_action),
        ],
    ).set_parse_action(infix_parse_action)
    return expr




def run_expression_str(parser, expression: str, ):
    full = parser.parseString(expression)
    ensure_equal(len(full), 1)
    merge: Merge = full[0]
    ensure_type(merge, Merge, "expected Merge at root")

    ctx = ParseCtx({ })
    run_merge(ctx, merge)

    # print(type(e))
    # print(repr(e))
    # ensure()
    print(full)


if __name__ == '__main__':
    test = [
        "Woop*10 + HUH*1",
        # "Woop*2 + HUH",
        # "Woop*2",
        # "(yas1800-SD_1.5) * 1 + DisEl*2 + Anyt ",
        # "(WOOP-(SD_1.5*2 + SD1.4*1)-(SD_1.5*2 + SD1.4*1)-WOOP)*1 + (DisEl)*2 + (Anyt *3 + HUH*3)*4",
        # "(T111)*2 + (M222*3 + M444*3)*4",
        # "T111*2 + (M222*3 + M444*3)*4 + (HMM-SD)*1",
        # "T111 + (M222*3 + M444*3)*4",
        # "(yas1800-SD15) * 1 + DisEl*2 + Anyt ",
    ]
    p = make_parser()
    for t in test:
        # print(t)
        # e = p.parseString(t)
        # ensure(len(e) == 1),al
        # e = e[0]
        # print(e)
        run_expression_str(p, t)
        pass
