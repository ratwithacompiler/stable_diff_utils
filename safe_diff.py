#!/usr/bin/env python3
#
# Copyright (c) 2022 RatWithAShotgun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Calculate total sum of difference between two checkpoints,
useful for making a good guess on which weights something was finetuned/dreamboothed on
by comparing abs sum and abs mean.
"""

import argparse
import builtins
import contextlib
import dataclasses
import logging as _logging
import os
import sys
import time

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from safe_load import statedict_convert_ema, statedict_strip_ema
from safe_multi_merge import torch_safe_load_dict_lazy, _precision_arg, LazyTensor

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Dict, Tuple
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection, Set
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

import torch

logger = _logging.getLogger(__name__)
IS_DEV = os.environ.get("DEV") == "1"


@dataclasses.dataclass()
class Timed():
    start: float
    end: Optional[float] = None
    secs: Optional[float] = None
    mins: Optional[float] = None
    hours: Optional[float] = None
    days: Optional[float] = None


@contextlib.contextmanager
def timed(print_format: Optional[str] = None, print = builtins.print):
    t = Timed(time.monotonic())
    yield t
    t.end = time.monotonic()
    t.secs = t.end - t.start
    t.mins = t.secs / 60
    t.hours = t.secs / 3600
    t.days = t.secs / (24 * 3600)
    if print_format is not None:
        output = print_format.format(
            [t.secs] * 128,
            secs = t.secs, s = t.secs,
            mins = t.mins, m = t.mins,
            hours = t.hours, h = t.hours,
            days = t.days, d = t.days,
        )
        print(output)


def state_dict_diff(
        one: dict,
        two: dict,
        tensor_diff_fn: Callable[[str, torch.Tensor, torch.Tensor], None],
        no_mixed: bool,
        no_differently_sized: bool,
        precision: str = "auto",
):
    if precision not in ("fp32", "fp16", "auto"):
        raise ValueError("invalid precision", precision)

    shared_keys = one.keys() & two.keys()
    # print(len(shared_keys))
    # print(sorted(shared_keys))

    expected_floats = { torch.float16, torch.float32 }
    use_precision = _precision_arg([one, two], precision, True)

    get_dtype = lambda tens_or_lazy: tens_or_lazy.dtype() if isinstance(tens_or_lazy, LazyTensor) else tens_or_lazy.dtype
    for key in shared_keys:
        maybe_lazy_tensors = [one[key], two[key]]
        dtypes = { get_dtype(i) for i in maybe_lazy_tensors }
        has_floats = dtypes & expected_floats
        non_floats = dtypes - expected_floats

        if has_floats and non_floats:
            # mix of floats and non floats, probably simply halfed
            if no_mixed:
                raise ValueError("dtype mix floats and non floats not allowed", key, dtypes)
            logger.info("ignoring key with mixed float/non float types: %r, %r", key, dtypes)
            continue

        tensors = [(i.load_copy() if isinstance(i, LazyTensor) else i) for i in maybe_lazy_tensors]
        if has_floats:
            tensors = [i.to(use_precision) for i in tensors]

        sizes = { i.storage().nbytes() for i in tensors }
        if len(sizes) > 1:
            if no_differently_sized:
                raise ValueError("tensors of various sizes, can't merge", key, sizes)

            logger.info("ignoring key with different size tensors: %r, %r", key, sizes)
            continue

        tensor_diff_fn(key, tensors[0], tensors[1])
        del tensors


def main(
        path1: str, path2: str,
        use_unsafe_torch_load: bool,
        ema_rename_require: bool, ema_rename_optional, ema_strip: bool,
):
    for i in (path1, path2):
        if not os.path.exists(i):
            raise FileNotFoundError(i)

    print(f"diffing {path1!r} - {path2!r}")
    if use_unsafe_torch_load:
        model1 = torch.load(path1)
        model2 = torch.load(path2)
    else:
        model1 = torch_safe_load_dict_lazy(path1, extended = True)
        model2 = torch_safe_load_dict_lazy(path2, extended = True)

    for mod in [model1, model2]:
        sd = mod["state_dict"]

        if ema_rename_require:
            print("replacing model keys with required ema model keys")
            statedict_convert_ema(sd, False, print_stats = True)
        elif ema_rename_optional:
            print("replacing model keys with ema model keys if present")
            statedict_convert_ema(sd, True, print_stats = True)

        if ema_strip:
            print("stripping ema model keys")
            statedict_strip_ema(sd, True)
        del sd

    # sum_cnt = 0
    # sums = torch.zeros(1024 * 64, dtype = torch.float64)
    sums = []

    def tens_diff(key: str, t1: torch.Tensor, t2: torch.Tensor):
        diff = torch.subtract(t1, t2)
        sum = torch.sum(diff, dtype = torch.float64)
        # print(key, "sum", sum.item())
        # print(key, "diff", diff)
        sums.append(sum.item())
        pass

    with timed("diffing took {secs:.2f} secs ({mins:.2f} mins)"):
        state_dict_diff(model1["state_dict"], model2["state_dict"], tens_diff, False, False)

    tsums = torch.tensor(sums)
    tsums_abs = torch.abs(tsums)

    total = sum(sums)
    std, mean = torch.std_mean(tsums)

    total_abs = sum(abs(i) for i in sums)
    std_abs, mean_abs = torch.std_mean(tsums_abs)

    print()
    print(f"Diff of {len(sums)} shared keys between: {path1!r} - {path2!r}:")
    print("diff sum: ", total)
    print("std dev : ", std.item())
    print("mean    : ", mean.item())

    print("abs sum     : ", total_abs)
    print("abs std dev : ", std_abs.item())
    print("abs mean    : ", mean_abs.item())


if __name__ == "__main__":
    def setup():
        parser = argparse.ArgumentParser()
        parser.add_argument("--use-unsafe-torch-load", action = "store_true")
        parser.add_argument("-e", "--ema-rename-try", action = "store_true",
                            help = "if ema keys present replace normal model keys with ema equivalent, ema keys not kept separately")
        parser.add_argument("--ema-rename", action = "store_true",
                            help = "replace normal model keys with ema equivalent, ema keys not kept separately, require ema keys")
        parser.add_argument("-E", "--ema-strip", action = "store_true",
                            help = "strip ema model keys")
        parser.add_argument("ckpt_1")
        parser.add_argument("ckpt_2")
        args = parser.parse_args()
        _logging.basicConfig(level = _logging.DEBUG)

        main(
            args.ckpt_1, args.ckpt_2, args.use_unsafe_torch_load,
            args.ema_rename, args.ema_rename_try, args.ema_strip,
        )


    setup()
