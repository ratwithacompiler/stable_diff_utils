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
#
# Uses pickletools and minimal implementation of pickling opcode processing to read torch pickled weights.
# Supports nothing expect basic python types (int, float, str, list, tuple, dicts; arbitrarily nested)
# and loading torch saved basic tensors.
# Skips everything else.

# dependencies: torch
# usage: python safe_load.py [--half] [--overwrite] path_to_input.ckpt  path_where_to_save.ckpt

import os.path
import argparse
import collections
import zipfile
import functools
from typing import Union

import torch
import pickletools

import logging as _logging

try:
    import safetensors.torch
except:
    safetensors = None

logger = _logging.getLogger(__name__)


def statedict_half(state_dict, print_stats: bool = False):
    halfed_cnt = 0
    halfed_bytes = 0
    total_bytes = 0

    for key, val in list(state_dict.items()):
        if not isinstance(val, torch.Tensor):
            continue

        total_bytes += val.element_size() * val.nelement()
        if val.dtype is torch.float32:  # avoid converting any ints to f16
            halfed = val.half()
            state_dict[key] = halfed
            halfed_cnt += 1
            halfed_bytes += halfed.element_size() * halfed.nelement()

    if print_stats:
        print(f"halfed {halfed_cnt} keys, {halfed_bytes} bytes, {halfed_bytes / 1024 ** 3:.2f} GB!")

    return (halfed_cnt, halfed_bytes, total_bytes)


def statedict_convert_ema(sd: dict, optional: bool, print_stats: bool = False, verbose: bool = False):
    update = { }
    stripped = { }
    missing = set()
    for key, val in sd.items():
        if key.startswith("model."):
            # replace each "model." key with the equivalent "model_ema." key
            ema_key = "model_ema." + key[6:].replace(".", "")
            try:
                update[key] = sd[ema_key]
                stripped[ema_key] = val
            except KeyError:
                if not optional:
                    raise
                missing.add(key)

    ema_keys = { i for i in sd.keys() if i.startswith("model_ema.") }
    ema_keys_kept = ema_keys - stripped.keys()

    for key in stripped:
        del sd[key]

    sd.update(update)

    if print_stats:
        verbose_str = ""
        if verbose:
            verbose_str = f"stripped: {sorted(stripped.keys())}, missed: {sorted(missing)}, kept: {sorted(ema_keys_kept)}."

        print(
            f"replaced {len(update)} model with ema keys. "
            f"ema keys: {len(missing)} missing, {len(ema_keys_kept)} non model kept. {verbose_str}"
        )

    return sd, stripped


def statedict_strip_ema(sd: dict, print_stats: bool = False, verbose: bool = False):
    stripped = { }
    for key, val in sd.items():
        if key.startswith("model_ema."):
            stripped[key] = val

    for key in stripped:
        del sd[key]

    if print_stats:
        if verbose:
            print(f"stripped {len(stripped)} ema model keys from state_dict: {sorted(stripped.keys())}")
        else:
            print(f"stripped {len(stripped)} ema model keys from state_dict")

    return sd, stripped


class IGNORED_REDUCE():
    def __init__(self, ignored_name):
        self.ignored_name = ignored_name

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"IGNORED_REDUCE({self.ignored_name!r})"


def _skip_kv(allow_int_keys: bool, key, val, label: str):
    if isinstance(key, str):
        if key.startswith("__"):
            logger.info("%s ignoring __ keyval %r: %r", label, key, val)
            return True
        return False

    if allow_int_keys and isinstance(key, int):
        return False

    raise ValueError("unsupported key", label, key)


def pickle_bytes_safe_load_dict(
        pickle_bytes: bytes, persistent_id_load_fn,
        reduce_fns_custom = None,
        reduce_fns_ignore_unknown = False,
        extended: bool = True,
):
    reduce_fns = {
        **{ 'collections OrderedDict': collections.OrderedDict },
        **(reduce_fns_custom or { }),
    }
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

        elif opcode.name in { "GET", "BINGET", "LONG_BINGET" }:
            stack.append(memo[arg])
            continue

        if opcode.name == "REDUCE":
            arg_tup = stack.pop()
            func_name = stack.pop()
            func = reduce_fns.get(func_name)
            if func is None:
                if reduce_fns_ignore_unknown:
                    logger.info("ignoring unkonwn reduce function %r with args %r", func_name, arg_tup)
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

        if extended and opcode.name == "BUILD":
            build_arg = stack.pop()
            last = stack[-1]
            if isinstance(last, dict) and isinstance(build_arg, dict):
                build_arg = { key: val for (key, val) in build_arg.items() if not _skip_kv(extended, key, val, "BUILD") }
                last.update(build_arg)
            else:
                logger.info("ignoring BUILD of object %r with args %r", last, build_arg)
            continue

        if opcode.name == "TUPLE":
            tup = tuple(stack_pop_until(markobject))
            stack.append(tup)
            continue

        if opcode.name == "APPENDS":
            values = stack_pop_until(markobject)
            target = stack[-1]
            if not isinstance(target, list):
                raise ValueError("expected list", type(target), target)
            target.extend(values)
            continue

        if opcode.name == "SETITEM":
            val = stack.pop()
            key = stack.pop()

            target = stack[-1]
            if not isinstance(target, dict):
                raise ValueError("expected settitems dict", type(target), target)

            if not _skip_kv(extended, key, val, "SETITEM"):
                target[key] = val
            continue

        if opcode.name == "SETITEMS":
            items = stack_pop_until(markobject)
            if len(items) % 2 != 0:
                raise ValueError("uneven SETITEMS key number", len(items), items)

            items = [(items[i], items[i + 1]) for i in range(0, len(items), 2)]
            set_map = dict(items)
            set_map = { key: val for (key, val) in set_map.items() if not _skip_kv(extended, key, val, "SETITEMS") }

            target = stack[-1]
            if not isinstance(target, dict):
                raise ValueError("expected settitems dict", type(target), target)

            target.update(set_map)
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
            "GLOBAL",
        }:
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


DTYPE_MAP = {
    "torch FloatStorage": (torch.float32, 4),
    "torch HalfStorage":  (torch.float16, 2),
    "torch IntStorage":   (torch.int32, 4),
    "torch LongStorage":  (torch.int64, 8),
}


def _build_tensor(zipfile, archive_name, storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if storage_offset or backward_hooks:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

    (storage, dtype_str, index, location, element_count) = storage
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    data_path = f"{archive_name}/data/{index}"
    data = zipfile.read(data_path)

    expected_size = element_count * dtype_size
    if len(data) != expected_size:
        raise ValueError("read unexpected amount of bytes",
                         len(data), expected_size, data_path, element_count, dtype_size)

    tensor = torch.frombuffer(data, dtype = dtype, requires_grad = requires_grad)
    return tensor.set_(tensor, storage_offset = 0, size = torch.Size(size), stride = stride)


def get_archive_name(zipfile: zipfile.ZipFile, required: bool, data_only: bool = True):
    names = set(zipfile.namelist())
    for file in zipfile.filelist:
        if "/" in file.filename:
            prefix = file.filename[:file.filename.index("/")]
            if not data_only:
                return prefix

            if f"{prefix}/data.pkl" in names:
                print(f"found {prefix=}")
                return prefix

    if required:
        raise ValueError("archive prefix not found")


def torch_safe_load_dict(model_path_or_zipfile: Union[str, zipfile.ZipFile], extended: bool = False):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = zipfile.ZipFile(model_path_or_zipfile)

    try:
        data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")
        archive_name = "archive"
    except KeyError:
        archive_name = get_archive_name(model_path_or_zipfile, True)
        data_pickle_bytes = model_path_or_zipfile.read(f"{archive_name}/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    build_tensor = functools.partial(_build_tensor, model_path_or_zipfile, archive_name)
    model = pickle_bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
        extended = extended,
    )

    return model


def _guess_filetype(path: str, default):
    path = str(path)
    if path.endswith(".safetensor") or path.endswith(".safetensors") or path.endswith(".st"):
        return "safetensors"

    if path.endswith(".pt") or path.endswith(".ckpt"):
        return "ckpt"

    return default


def main(input_path: str, output_path: str, overwrite: bool, half: bool, extended: bool,
         ema_rename_require: bool, ema_rename_optional, ema_strip: bool,
         set_times: bool, use_tmpfile: bool, fixed_write_filetype: str):
    if not overwrite and os.path.exists(output_path):
        raise ValueError(f"output_file path exists already, overwriting disabled {output_path!r}")

    print(f"loading {input_path!r}")

    filetype = _guess_filetype(input_path, "ckpt")
    if filetype == "ckpt":
        model = torch_safe_load_dict(input_path, extended)
    elif filetype == "safetensors":
        sd = safetensors.torch.load_file(input_path)
        # if "state_dict" in sd:
        #     model = sd
        # else:
        #     model = { "state_dict": sd }
        model = { "state_dict": sd }
        del sd
    else:
        raise ValueError("invalid filetype", filetype)

    try:
        sd = model["state_dict"]
    except:
        if "model.diffusion_model.input_blocks.0.0.weight" in model:
            print("loaded direct state_dict")
            sd = model
        else:
            raise

    if half:
        print("halfing")
        statedict_half(sd, True)

    if ema_rename_require:
        print("replacing model keys with required ema model keys")
        statedict_convert_ema(sd, False, print_stats = True)
    elif ema_rename_optional:
        print("replacing model keys with ema model keys if present")
        statedict_convert_ema(sd, True, print_stats = True)

    if ema_strip:
        print("stripping ema model keys")
        statedict_strip_ema(sd, True)

    model = { "state_dict": sd }

    filetype = fixed_write_filetype or _guess_filetype(output_path, "ckpt")

    if use_tmpfile:
        write_path = f"{output_path}.tmp"
        mode = "wb"
        print(f"writing to tmp file {write_path!r} as {filetype}")
    else:
        write_path = output_path
        mode = "wb" if overwrite else "xb"
        print(f"writing to {output_path!r} as {filetype}, overwrite={overwrite}")

    if filetype == "ckpt":
        with open(write_path, mode) as out_file:
            torch.save(model, out_file)
    elif filetype == "safetensors":
        # non_tens = { k: v for (k, v) in sd.items() if not isinstance(v, torch.Tensor) }
        # print("non_tens", non_tens.keys(), non_tens)
        sd = model.get("state_dict") or model
        if "state_dict" in sd and sd["state_dict"] == { }:
            print("fixed removed empty {} state_dict inside state_dict")
            del sd["state_dict"]

        safetensors.torch.save_file(sd, write_path)
    else:
        raise ValueError("invalid filetype", filetype)

    if use_tmpfile:
        if not overwrite and os.path.exists(output_path):
            raise ValueError(f"output_file path exists, didn't before, overwriting disabled {output_path!r}")

        assert write_path != output_path
        os.rename(write_path, output_path)
        print(f"moved to to {output_path!r}")

    if set_times:
        cur = os.stat(input_path)
        print(f"setting access/modified times of {input_path!r} on {output_path!r}", (cur.st_atime_ns, cur.st_mtime_ns))
        os.utime(output_path, ns = (cur.st_atime_ns, cur.st_mtime_ns))

    print("done")


if __name__ == "__main__":
    def setup():
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("output_file")
        parser.add_argument("-s", "--simple", action = "store_true", help = "no BUILD, int keys")
        parser.add_argument("-o", "--overwrite", action = "store_true")
        parser.add_argument("-H", "--half", action = "store_true")

        format_group = parser.add_mutually_exclusive_group()
        format_group.add_argument("-C", "--write-ckpt", action = "store_true")
        format_group.add_argument("-S", "--write-safetensors", action = "store_true")

        parser.add_argument("-e", "--ema-rename-try", action = "store_true", help = "if ema keys present replace normal model keys with ema equivalent, ema keys not kept separately")
        parser.add_argument("--ema-rename", action = "store_true", help = "replace normal model keys with ema equivalent, ema keys not kept separately, require ema keys")
        parser.add_argument("-E", "--ema-strip", action = "store_true", help = "strip ema model keys")
        parser.add_argument("-t", "--times", action = "store_true", help = "set same access/modified time on output file as on input file")

        parser.add_argument("-N", "--no-tempfile", action = "store_true", help = "write to output file directly, don't use tempfile and rename")

        # parser.add_argument("-S", "--strip", choices = ["ema", "non_ema"])
        args = parser.parse_args()
        _logging.basicConfig(level = _logging.DEBUG)

        fixed_write_filetype = None
        if args.write_ckpt:
            fixed_write_filetype = "ckpt"
        if args.write_safetensors:
            fixed_write_filetype = "safetensors"

        main(args.input_file, args.output_file, args.overwrite, args.half, not args.simple, args.ema_rename, args.ema_rename_try, args.ema_strip,
             args.times, not args.no_tempfile, fixed_write_filetype)


    setup()
