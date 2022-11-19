#!/usr/bin/env python3

import sys
from dataclasses import dataclass

import sys

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Tuple, Type, Dict, Set
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

from type_utils import ensure_type, ensure, ensure_equal


class Tens():
    pass


@dataclass()
class Var(Tens):
    name: str

    # def __repr__(self):
    #     # return f"V({self.name})"
    #     return self.name


@dataclass()
class Factored():
    item: Tens
    factor: Optional[Union[int, float]] = None

    # def __repr__(self):
    #     # return f"Fac{{{self.item}*{self.factor}}}"
    #     return f"Fac{self.factor}{{{self.item}}}"


@dataclass()
class Merge(Tens):
    items: List[Factored]

    def factor_total(self):
        return sum(i.factor for i in self.items)

    # def __repr__(self):
    #     return f" Merge[ {' + '.join(repr(i) for i in self.items)} ]"


@dataclass()
class Minus(Tens):
    left: Tens
    right: Tens

    # def __repr__(self):
    #     return f" {self.left!r} - {self.right!r} "


# @dataclass()
# class Group(Tens):
#     item: Tens


def var_parse_action(section, pos, res):
    # print("var_parse_action", (pos, res))
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
        # if isinstance(i, Merge):
        #     for a in i.items:
        #         ensure_type(a, Factored, "plus expanded arg must have factor", pos, type(i), i, full)
        #     use.extend(i.items)
        # else:
        if isinstance(i, Tens):
            i = Factored(i, 1)
        ensure_type(i, Factored, "plus arg must have factor", pos, type(i), i, full)
        use.append(i)
    return Merge(use)


def infix_parse_action(section, pos, full):
    # print("infix_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    # res = full[0]
    # return Group(res)
    return full


def make_parser(enablePackrat: Optional[bool] = None):
    import pyparsing as pyp
    from pyparsing import ParserElement, pyparsing_common, Word

    if enablePackrat:
        ParserElement.enablePackrat()

    integer = pyparsing_common.integer
    simple_positive_float = (
        pyp.Regex(r"\d+\.\d*")
        .set_name("basic float")
        .set_parse_action(pyparsing_common.convert_to_float)
    )
    variable = Word(pyp.alphas, pyp.alphanums + "._").set_parse_action(var_parse_action)
    operand = simple_positive_float | integer | variable

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


def merge_to_str(merge):
    def _exp_get(tens: Tens):
        if isinstance(tens, Merge):
            return "(" + _exp_run_merge(tens) + ")"
        if isinstance(tens, Var):
            return tens.name
        if isinstance(tens, Minus):
            return "(" + _exp_run_minus(tens) + ")"
        raise ValueError("unsupported tens", tens)

    def _exp_run_minus(minus: Minus):
        left = _exp_get(minus.left)
        right = _exp_get(minus.right)
        return f"{left}-{right}"

    def _exp_run_merge(merge: Merge):
        factor_total = merge.factor_total()

        parts = []
        for i in merge.items:
            tens = _exp_get(i.item)
            fact = i.factor / factor_total
            fact = f"{fact:.3f}".rstrip("0").rstrip(".")
            parts.append(f"{tens}*{fact}")
        return "_+_".join(parts)

    return _exp_run_merge(merge)


def merge_vars(merge: Merge) -> Set[str]:
    seen_vars = set()

    def _exp_get(tens: Tens):
        if isinstance(tens, Merge):
            _exp_run_merge(tens)
        elif isinstance(tens, Var):
            seen_vars.add(tens.name)
        elif isinstance(tens, Minus):
            _exp_run_minus(tens)
        else:
            raise ValueError("unsupported tens", tens)

    def _exp_run_minus(minus: Minus):
        _exp_get(minus.left)
        _exp_get(minus.right)

    def _exp_run_merge(merge: Merge):
        for i in merge.items:
            _exp_get(i.item)

    _exp_run_merge(merge)
    return seen_vars


def parse_merge_expression(parser, expression: str) -> Merge:
    full = parser.parse_string(expression, parse_all = True)
    ensure_equal(len(full), 1)
    merge = full[0]
    ensure_type(merge, Merge, "expected Merge at root")
    return merge
