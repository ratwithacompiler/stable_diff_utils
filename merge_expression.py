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

MERGE_CHAR = "+"
ADD_CHAR = "<"
MINUS_CHAR = "-"
FACTOR_CHAR = "@"


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
    factor: Union[int, float]
    factor_str: Optional[str] = None

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


@dataclass()
class Add(Tens):
    left: Tens
    right: Tens

    def __repr__(self):
        return f" {self.left!r} < {self.right!r} "


@dataclass()
class RunExp():
    item: Union[Add, Merge]


# @dataclass()
# class Group(Tens):
#     item: Tens


def var_parse_action(section, pos, res):
    # print("var_parse_action", (pos, res))
    ensure_equal(len(res), 1)
    ensure_type(res[0], str)
    return Var(res[0])


def factor_parse_action(section, pos, full):
    # print("factor_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]
    # for pos, i in enumerate(res):
    #     print(pos, i)
    ensure_equal(len(res), 3, "factor only allowed once for variable/expression", len(res), pos, res)
    ensure_type(res[0], Tens, "left factor op must be Tensor", res[0], full)
    ensure_equal(res[1], FACTOR_CHAR)

    num_str = res[2]
    if isinstance(num_str, int):
        num = num_str
        num_str = str(num_str)
    else:
        num = float(num_str) if num_str.count(".") else int(num_str)
    ensure_type(num_str, str)
    return Factored(res[0], num, num_str)


def minus_parse_action(section, pos, full):
    # print("minus_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]
    ensure_equal(len(res), 3)

    for pos, i in enumerate(res):
        if i == MINUS_CHAR:
            continue
        # print(pos, i)
        ensure_type(i, Tens, "minus argument must be tensor", pos, type(i), i)

    # ensure(len(res) == 3, "minus only allowed once for variable/expression", len(res), pos, res)
    return Minus(res[0], res[2])


def add_parse_action(section, pos, full):
    # print("add_parse_action", (pos, full))
    ensure_equal(len(full), 1)
    res = full[0]
    ensure_equal(len(res), 3)

    for pos, i in enumerate(res):
        if i == ADD_CHAR:
            continue
        # print(pos, i)
        ensure_type(i, Tens, "add argument must be tensor", pos, type(i), i)

    # ensure(len(res) == 3, "minus only allowed once for variable/expression", len(res), pos, res)
    return Add(res[0], res[2])


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


def make_parser(enablePackrat: Optional[bool] = True):
    import pyparsing as pyp
    from pyparsing import ParserElement, pyparsing_common, Word

    if enablePackrat:
        ParserElement.enablePackrat()

    integer = pyparsing_common.integer
    simple_positive_float = (
        pyp.Regex(r"\d+\.\d*")
        .set_name("basic float")
        # .set_parse_action(pyparsing_common.convert_to_float)
    )
    variable = Word(pyp.alphas, pyp.alphanums + "._").set_parse_action(var_parse_action)
    operand = simple_positive_float | integer | variable

    factor_op = pyp.oneOf(FACTOR_CHAR)
    merge_op = pyp.oneOf(MERGE_CHAR)
    minus_op = pyp.oneOf(MINUS_CHAR)
    add_op = pyp.oneOf(ADD_CHAR)

    # minusop = pyp.oneOf("-")
    # plusop = pyp.oneOf("+")

    expr = pyp.infix_notation(
        operand,
        [
            (factor_op, 2, pyp.opAssoc.LEFT, factor_parse_action),
            (minus_op, 2, pyp.opAssoc.LEFT, minus_parse_action),
            (add_op, 2, pyp.opAssoc.LEFT, add_parse_action),
            (merge_op, 2, pyp.opAssoc.LEFT, plus_parse_action),
        ],
    ).set_parse_action(infix_parse_action)
    return expr


def runexp_to_str(runexp: RunExp, use_original_factor: bool):
    def _exp_get(tens: Tens):
        if isinstance(tens, Merge):
            return "(" + _exp_run_merge(tens) + ")"
        if isinstance(tens, Var):
            return tens.name
        if isinstance(tens, Minus):
            return "(" + _exp_run_minus(tens) + ")"
        if isinstance(tens, Add):
            return "(" + _exp_run_add(tens) + ")"
        raise ValueError("unsupported tens", tens)

    def _exp_run_minus(minus: Minus):
        left = _exp_get(minus.left)
        right = _exp_get(minus.right)
        return f"{left}{MINUS_CHAR}{right}"

    def _exp_run_add(add: Add):
        left = _exp_get(add.left)
        right = _exp_get(add.right)
        return f"{left}{ADD_CHAR}{right}"

    def _exp_run_merge(merge: Merge):
        factor_total = merge.factor_total()

        parts = []
        for i in merge.items:
            tens = _exp_get(i.item)
            if use_original_factor:
                fact = i.factor_str
            else:
                fact = i.factor / factor_total
                fact = f"{fact:.3f}".rstrip("0").rstrip(".")

            if fact is None:
                parts.append(tens)
            else:
                parts.append(f"{tens}{FACTOR_CHAR}{fact}")

        return f" {MERGE_CHAR} ".join(parts)

    if isinstance(runexp.item, Merge):
        return _exp_run_merge(runexp.item)
    elif isinstance(runexp.item, Add):
        return _exp_run_add(runexp.item)
    raise ValueError("invalid runexp type", runexp.item)


def runexp_vars(runexp: RunExp) -> Set[str]:
    seen_vars = set()

    def _exp_get(tens: Tens):
        if isinstance(tens, Merge):
            _exp_run_merge(tens)
        elif isinstance(tens, Var):
            seen_vars.add(tens.name)
        elif isinstance(tens, Minus):
            _exp_run_minus(tens)
        elif isinstance(tens, Add):
            _exp_run_add(tens)
        else:
            raise ValueError("unsupported tens", tens)

    def _exp_run_minus(minus: Minus):
        _exp_get(minus.left)
        _exp_get(minus.right)

    def _exp_run_add(add: Add):
        _exp_get(add.left)
        _exp_get(add.right)

    def _exp_run_merge(merge: Merge):
        for i in merge.items:
            _exp_get(i.item)

    if isinstance(runexp.item, Merge):
        _exp_run_merge(runexp.item)
    elif isinstance(runexp.item, Add):
        _exp_run_add(runexp.item)
    else:
        raise ValueError("invalid runexp type", runexp.item)
    return seen_vars


def parse_run_expression(parser, expression: str) -> RunExp:
    full = parser.parse_string(expression, parse_all = True)
    ensure_equal(len(full), 1)
    exp = full[0]
    ensure_type(exp, (Merge, Add), "expected Merge or Add at root")
    return RunExp(exp)
