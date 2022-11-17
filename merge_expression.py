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


def _run_expression_str(parser, expression: str, ):
    print("\n" * 3)
    print("checking", repr(expression))
    full = parser.parse_string(expression, parse_all = True)
    ensure_equal(len(full), 1)
    merge: Merge = full[0]
    ensure_type(merge, Merge, "expected Merge at root")
    print("  ", merge)
    print(merge_to_str(merge))
    return

    @dataclass()
    class ParseCtx():
        var_map: Dict

    def _get(ctx: ParseCtx, tens: Tens):
        if isinstance(tens, Merge):
            return run_merge(ctx, tens)

        if isinstance(tens, Var):
            # return ctx.var_map[tens.name]
            return f"Var({tens.name})"

        if isinstance(tens, Minus):
            return run_minus(ctx, tens)

        raise ValueError("unsupported tens", tens)

    def run_minus(ctx: ParseCtx, minus: Minus):
        left = _get(ctx, minus.left)
        right = _get(ctx, minus.right)
        return f"Minus({left}-{right})"

    def run_merge(ctx: ParseCtx, merge: Merge):
        factors_total = sum(i.factor for i in merge.items)

        base = _get(ctx, merge.items[0].item) * max(1, round(merge.items[0].factor))
        for i in merge.items[1:]:
            tens = _get(ctx, i.item)
            base += tens * max(1, round(i.factor))

        print("base", base)
        return base

    ctx = ParseCtx({ })
    run_merge(ctx, merge)

    # print(type(e))
    # print(repr(e))
    # ensure()
    print(full)


if __name__ == '__main__':
    test = [
        # "Any3 + (Gibli-SD14) + (DisEl-SD15) + Otherr*0.7 + Yas1800*1.3",
        # "Any3 + (Gibli-SD14) + (DisEl-SD15) + (Otherr*0.7 + Yas1800*1.3)",
        # "Any3 +  (Otherr*0.7 + Yas1800*1.3 + HUH)",
        "Any3 + (Gibli-SD14)*4 + (Mre+HUH+ (a-b))",
        # "Any3 + BBB + CCC + (DDDD + EEE + EE2222) + FFFF + GGGG + HHHH",
        # "Woop*10 + HUH*1",
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
        _run_expression_str(p, t)
        pass
