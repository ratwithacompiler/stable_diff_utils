#!/usr/bin/env python3

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


# Uses pickletools and minimal implementation of pickling opcode processing to read torch pickled weights.
# Supports nothing expect basic python types (int, float, list, tuple, dicts; arbitrarily nested)
# and loading torch saved basic tensors.
# Skips everything else.

# dependencies: torch
# usage: python safe_load.py [--prune] [--overwrite] path_to_input.ckpt  path_where_to_save.ckpt

import os.path
import sys
import argparse
import collections
import zipfile
from typing import Union

import torch
import pickletools


def eprint(*args, **kwargs):
    print(*args, **kwargs, file = sys.stderr)


def statedict_prune(state_dict, print_stats: bool = True):
    halfed_cnt = 0
    halfed_bytes = 0

    for key, val in list(state_dict.items()):
        if val.dtype is torch.float32:  # avoid converting any ints to f16
            halfed = val.half()
            state_dict[key] = halfed
            halfed_cnt += 1
            halfed_bytes += halfed.element_size() * halfed.nelement()

    if print_stats:
        print(f"pruned {halfed_cnt} keys, {halfed_bytes} bytes, {halfed_bytes / 1024 ** 3:.2f} GB!")

    return (halfed_cnt, halfed_bytes)


class IGNORED_REDUCE():
    def __init__(self, ignored_name):
        self.ignored_name = ignored_name

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"IGNORED_REDUCE({self.ignored_name!r})"


def bytes_safe_load_dict(
        pickle_bytes: bytes, persistent_id_load_fn,
        reduce_fns_custom = None,
        reduce_fns_ignore_unknown = False,
):
    reduce_fns = {
        **{ 'collections OrderedDict': collections.OrderedDict },
        **(reduce_fns_custom or { }),
    }
    # print("reduce_fns", reduce_fns)

    stack = []
    memo = { }

    markobject = pickletools.markobject

    def stack_pop_until(end):
        items = []
        while True:
            item = stack.pop()
            if item is end:
                break
            items.append(item)
        return list(reversed(items))

    for opcode, arg, pos in pickletools.genops(pickle_bytes):
        # print((opcode.name, arg, pos))

        if opcode.name == "PROTO":
            # print("ignoring proto", opcode.name)
            continue

        if opcode.name == "STOP":
            break

        if opcode.name == "EMPTY_DICT":
            stack.append({ })
            continue

        if opcode.name in { "BINPUT", "LONG_BINPUT" }:
            # print("MEMO set", (opcode.arg, stack[-1]))
            memo[arg] = stack[-1]
            continue

        elif opcode.name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
            continue

        if opcode.name == "REDUCE":
            arg_tup = stack.pop()
            func_name = stack.pop()
            func = reduce_fns.get(func_name)
            if func is None:
                if reduce_fns_ignore_unknown:
                    eprint(f"ignoring unkonwn reduce function {func_name!r} with args {arg_tup!r}")
                    stack.append(IGNORED_REDUCE(str(func_name)))
                    continue
                raise ValueError("unsupported reduce function", repr(func_name), arg_tup)

            # print("REDUCE", (func, arg_tup))
            item = func(*arg_tup)
            stack.append(item)
            continue

        if opcode.name == "EMPTY_LIST":
            stack.append([])
            continue

        if opcode.name == "EMPTY_TUPLE":
            stack.append(tuple())
            continue

        if opcode.name == "MARK":
            stack.append(markobject)
            continue

        if opcode.name == "NONE":
            stack.append(None)
            continue

        if opcode.name == "NEWTRUE":
            stack.append(True)
            continue

        if opcode.name == "NEWFALSE":
            stack.append(False)
            continue

        if opcode.name == "BUILD":
            build_arg = stack.pop()
            last = stack[-1]
            eprint(f"ignoring BUILD of object {last!r} with args {build_arg!r}")
            continue

        if opcode.name == "TUPLE":
            tup = tuple(stack_pop_until(markobject))
            stack.append(tup)
            continue

        if opcode.name == "SETITEM":
            val = stack.pop()
            key = stack.pop()

            if key.startswith("__"):
                eprint(f"SETITEMS ignoring __ keyval {key!r}: {val!r}")
                continue

            stack[-1][key] = val
            continue

        if opcode.name == "SETITEMS":
            items = stack_pop_until(markobject)
            if len(items) % 2 != 0:
                raise ValueError("uneven SETITEMS key number", len(items), items)

            items = [(items[i], items[i + 1]) for i in range(0, len(items), 2)]
            map = dict(items)
            use_map = { }
            for k, v in map.items():
                if k.startswith("__"):
                    eprint(f"SETITEMS ignoring __ keyval {k!r}: {v!r}")
                    continue
                use_map[k] = v

            stack[-1].update(use_map)
            continue

        if opcode.name == "TUPLE1":
            stack[-1] = tuple(stack[-1:])
            continue

        if opcode.name == "TUPLE2":
            stack[-2:] = [tuple(stack[-2:])]
            continue

        if opcode.name == "TUPLE3":
            stack[-3:] = [tuple(stack[-3:])]
            continue

        if opcode.name == "BINPERSID":
            persistent_id = stack.pop()
            stack.append(persistent_id_load_fn(persistent_id))
            continue

        if opcode.name == "APPEND":
            item = stack.pop()
            the_list = stack[-1]
            the_list.append(item)

            continue

        if opcode.name in {
            "BINUNICODE",
            "BININT1", "BININT2", "BININT", "LONG", "LONG1", "LONG4",
            "BINFLOAT",
            "GLOBAL" }:
            # print(f"OPT {opcode.name!r} pushing {arg !r}")
            stack.append(arg)
            continue

        raise ValueError("unsupported opcode", opcode.name, opcode)

    if len(stack) != 1:
        raise ValueError("invalid stack left", len(stack), stack)

    last = stack[0]
    if not isinstance(last, dict):
        raise ValueError("invalid last stack item not dict", type(last), last)

    return last


def torch_safe_load_dict(model_path_or_zipfile: Union[str, zipfile.ZipFile]):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = zipfile.ZipFile(model_path_or_zipfile)

    DTYPE_MAP = {
        "torch FloatStorage": (torch.float32, 4),
        "torch HalfStorage":  (torch.float16, 2),
        "torch IntStorage":   (torch.int32, 4),
        "torch LongStorage":  (torch.int64, 8),
    }

    data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    def build_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        if storage_offset or backward_hooks:
            raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

        (storage, dtype_str, index, location, element_count) = storage
        assert storage == "storage"

        dtype, dtype_size = DTYPE_MAP[dtype_str]
        data_path = f"archive/data/{index}"
        data = model_path_or_zipfile.read(data_path)

        expected_size = element_count * dtype_size
        if len(data) != expected_size:
            raise ValueError("read unexpected amount of bytes",
                             len(data), expected_size, data_path, element_count, dtype_size)

        tensor = torch.frombuffer(data, dtype = dtype, requires_grad = requires_grad)
        return tensor.set_(tensor, storage_offset = 0, size = torch.Size(size), stride = stride)

    model = bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
    )

    return model


def main(input_path: str, output_path: str, overwrite: bool, half: bool):
    if not overwrite and os.path.exists(output_path):
        raise ValueError(f"output_file path exists already, overwriting disabled {output_path!r}")

    print(f"loading {input_path!r}")
    model = torch_safe_load_dict(input_path)
    sd = model["state_dict"]

    if half:
        print("pruning")
        statedict_prune(sd, True)

    model = { "state_dict": sd }

    print(f"writing to {output_path!r}, overwrite={overwrite}")
    with open(output_path, "wb" if overwrite else "xb") as out_file:
        torch.save(model, out_file)

    print("done")


if __name__ == "__main__":
    def setup():
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("output_file")
        parser.add_argument("-o", "--overwrite", action = "store_true")
        parser.add_argument("-p", "--prune", action = "store_true")
        args = parser.parse_args()
        main(args.input_file, args.output_file, args.overwrite, args.prune)


    setup()
