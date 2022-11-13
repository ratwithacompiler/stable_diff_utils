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

import argparse
import dataclasses
import functools
import io
import itertools
import logging as _logging
import os
import sys
import zipfile
from zipfile import ZipFile

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Dict
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

from safe_load import pickle_bytes_safe_load_dict, DTYPE_MAP, statedict_convert_ema, statedict_strip_ema, _build_tensor

import torch

logger = _logging.getLogger(__name__)
IS_DEV = os.environ.get("DEV") == "1"


@dataclasses.dataclass()
class Input():
    path: str
    zip_file: Optional[ZipFile] = None
    model: Optional[dict] = None

    def open(self):
        if self.zip_file is not None:
            raise ValueError("already open", self.path)

        self.zip_file = ZipFile(self.path, mode = "r")

    def state_dict(self) -> dict:
        return self.model["state_dict"]


@dataclasses.dataclass()
class Output():
    path: Optional[str]
    configs: List[Any]

    write_path: Optional[str] = None
    zip_file: Optional[ZipFile] = None
    model: Optional[dict] = None

    def open(self, mode: str, compression = zipfile.ZIP_STORED, compresslevel = None):
        if self.zip_file is not None:
            raise ValueError("already open", self.path)

        self.zip_file = ZipFile(self.write_path, mode = mode, compression = compression, compresslevel = compresslevel)

    def open_buffer(self, mode: str, compression = zipfile.ZIP_STORED, compresslevel = None):
        if self.zip_file is not None:
            raise ValueError("already open", self.path)

        buffer = io.BytesIO()
        self.zip_file = ZipFile(buffer, mode = mode, compression = compression, compresslevel = compresslevel)

    def close(self):
        self.zip_file.close()
        self.zip_file = None


@dataclasses.dataclass()
class LazyTensor():
    _ctx: Any
    _zip_file: ZipFile
    _storage_tup: tuple
    _data_path: str

    _storage_offset: int
    _size: torch.Size
    _stride: Union[tuple, int]
    _requires_grad: bool

    tensor: Optional[torch.Tensor] = None

    def load(self, reload = True):
        if not reload and self.tensor is not None:
            raise ValueError("tensor loaded already, reload not allowed")

        self.tensor = t = self.load_copy()
        return t

    def unload(self):
        self.tensor = None

    def load_copy(self):
        return _build_tensor(
            self._zip_file, self._storage_tup, self._storage_offset, self._size,
            self._stride, self._requires_grad, None,
        )


def _build_lazy_tensor(ctx, zipfile: ZipFile, storage_tup, storage_offset, size, stride, requires_grad, backward_hooks):
    if storage_offset or backward_hooks:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

    (storage, dtype_str, index, location, element_count) = storage_tup
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    data_path = f"archive/data/{index}"
    data_size = zipfile.getinfo(data_path).file_size

    expected_size = element_count * dtype_size
    if data_size != expected_size:
        raise ValueError("read unexpected amount of bytes",
                         data_size, expected_size, data_path, element_count, dtype_size)

    return LazyTensor(ctx, zipfile, storage_tup, data_path, storage_offset, torch.Size(size), stride, requires_grad)


def torch_safe_load_dict_lazy(model_path_or_zipfile: Union[str, ZipFile], extended: bool = False, tensor_ctx = None):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = ZipFile(model_path_or_zipfile)

    data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    build_tensor = functools.partial(_build_lazy_tensor, tensor_ctx, model_path_or_zipfile)
    model = pickle_bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
        extended = extended,
    )

    return model


def merge_weighted(ctx, tensors: List[torch.Tensor], factors: List[Union[int, float]]) -> torch.Tensor:
    assert len(tensors) == len(factors)
    total = sum(factors)
    factors = [i / total for i in factors]

    base: torch.Tensor = tensors[0] * factors[0]
    for pos, t in enumerate(tensors):
        if not pos:
            continue
        base = base + tensors[pos] * factors[pos]
    return base


def state_dicts_merge(
        state_dicts: List[dict],
        merge_contexts: List[Any],
        require_all: bool,
        merge_tensors_fn: Callable[[str, Any, List[Optional[torch.Tensor]]], Optional[torch.Tensor]],
        merge_others_fn: Callable[[str, List[Optional[Any]]], Any],
) -> List[Dict]:
    state_dicts = list(state_dicts)
    all_keys = set(itertools.chain(*state_dicts))

    print(len(all_keys))
    # print(sorted(all_keys))

    if IS_DEV:
        print("IS_DEV cutting only!!!")
        only = [
            'model.diffusion_model.input_blocks.8.0.out_layers.0.weight', 'model.diffusion_model.input_blocks.8.0.out_layers.3.bias', 'model.diffusion_model.input_blocks.8.0.out_layers.3.weight',
            'model.diffusion_model.input_blocks.8.1.norm.bias', 'model.diffusion_model.input_blocks.8.1.norm.weight', 'model.diffusion_model.input_blocks.8.1.proj_in.bias',
            'model.diffusion_model.input_blocks.8.1.proj_in.weight', 'model.diffusion_model.input_blocks.8.1.proj_out.bias', 'model.diffusion_model.input_blocks.8.1.proj_out.weight',
            'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight', 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias',
            'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight', 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight',
            'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight', 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight',
            'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias', 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight',
            'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight', 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight'
        ]
        only = set(only)
        all_keys = all_keys & only

    merge_contexts = list(merge_contexts)
    new_sd_tups = [(i, { }) for i in merge_contexts]
    for key in all_keys:
        lazy_tensors = []
        others = []
        for sd in state_dicts:
            val = sd.get(key)
            if isinstance(val, LazyTensor):
                lazy_tensors.append(val)
            else:
                if val is not None:
                    print("ignoring non tensor key", key, type(val))
                lazy_tensors.append(None)
                others.append(val)

        if len(others) == len(lazy_tensors):
            for merge_ctx, new_sd in new_sd_tups:
                new_sd[key] = merge_others_fn(key, others)
            continue

        if require_all and None in lazy_tensors:
            missing = [pos for (pos, i) in enumerate(lazy_tensors) if i is None]
            raise ValueError(f"tensor missing from {len(missing)} state_dicts", key, missing)

        # print(key, lazy_tensors)
        tensors = []
        for pos, i in enumerate(lazy_tensors):
            tensors.append(i.load_copy() if i is not None else None)

        for merge_ctx, new_sd in new_sd_tups:
            new_sd[key] = merge_tensors_fn(key, merge_ctx, tensors)

        del tensors

    return [new_sd for merge_ctx, new_sd in new_sd_tups]


def _merge_others(key: str, ctx, vals: List[Any]):
    print("merge_others!!!", key, len(vals), vals)
    return vals[0]


def _tensors_configs_filter(key: str, ctx: Output, inputs: List[Input], maybe_tensors: List[Optional[torch.Tensor]]):
    # filter out None tensors and return with list of corresponding configs
    assert len(maybe_tensors) == len(inputs)
    assert len(maybe_tensors) == len(ctx.configs)

    with_tensor = itertools.zip_longest(maybe_tensors, ctx.configs)
    with_tensor = [(t, c) for (t, c) in with_tensor if t is not None]
    if not with_tensor:
        raise ValueError("no tensors", maybe_tensors, with_tensor)

    tensors = [t for (t, c) in with_tensor]
    configs = [c for (t, c) in with_tensor]
    return tensors, configs


def _to_inputs(inputs: List[Union[Input, str]]) -> List[Input]:
    use_inputs = []
    for i in inputs:
        if not isinstance(i, Input):
            i = Input(i)
            i.open()
        use_inputs.append(i)
    return use_inputs


def inputs_outputs_merge_in_memory(
        inputs: List[Union[Input, str]],
        configs: List[Any],
        merge_fn: Callable,
        require_all: bool = False,
        merge_others = _merge_others,
):
    inputs = _to_inputs(inputs)

    def merge_tensors(key: str, ctx: Output, maybe_tensors: List[Optional[torch.Tensor]]):
        tensors, factors = _tensors_configs_filter(key, ctx, inputs, maybe_tensors)
        # print("merge_tensors!!!", key, sum(factors), factors, len(tensors), len(maybe_tensors))
        if len(tensors) == 1:
            return torch.clone(tensors[0])
        return merge_fn(ctx, tensors, factors)

    sds = [i.state_dict() for i in inputs]
    outputs = [Output(None, i) for i in configs]
    new_sds = state_dicts_merge(sds, outputs, require_all, merge_tensors, merge_others)
    return new_sds


def _create_persistent_id_fns():
    # Creates a torch _save persistent_id function for pickling torch Tensors.
    # Each torch persistent_id function has its own local incrementing counter
    # used for tensor storage key generation.

    import torch.serialization
    if torch.serialization.DEFAULT_PROTOCOL != 2:
        raise ValueError("expected torch.serialization.DEFAULT_PROTOCOL == 2, got ", torch.serialization.DEFAULT_PROTOCOL)

    class FoundException(Exception):
        def __init__(self, persistent_id):
            self.persistent_id = persistent_id

    class CatchingPickler():
        def __init__(self, data, protocol = None):
            pass

        def __setattr__(self, key, value):
            if value != "ignore":
                raise FoundException(value)

    class PickleModule():
        Pickler = CatchingPickler

    try:
        torch.serialization._save(None, None, PickleModule, 2)
    except FoundException as ex:
        return ex.persistent_id


class _OutputWriter():
    def __init__(self, output: Output):
        self.output = output
        self.persistent_id = _create_persistent_id_fns()


def inputs_outputs_merge_torch_zip_stream(
        inputs: List[Union[Input, str]],
        outputs: List[Output],
        merge_fn: Callable,
        require_all: bool = False,
        merge_others = _merge_others,
):
    import ctypes, pickle
    import torch._utils

    @dataclasses.dataclass
    class _PersId():
        persistent_id: Any

    @dataclasses.dataclass
    class _ReduceRes():
        reduce_res: Any

        def __reduce__(self):
            return self.reduce_res

    inputs = _to_inputs(inputs)
    output_writers = { id(i): _OutputWriter(i) for i in outputs }
    print(len(output_writers), output_writers)

    def merge_tensors(key: str, ctx: Output, maybe_tensors: List[Optional[torch.Tensor]]):
        writer: _OutputWriter = output_writers[id(ctx)]

        tensors, factors = _tensors_configs_filter(key, ctx, inputs, maybe_tensors)
        # print("merge_tensors!!!", key, sum(factors), factors, len(tensors), len(maybe_tensors))
        if len(tensors) == 1:
            tensor = tensors[0]
        else:
            tensor = merge_fn(ctx, tensors, factors)

        pers_id_tup = writer.persistent_id(tensor.storage())
        if pers_id_tup[0] != "storage":
            raise ValueError("expected storage as persistent id 0", pers_id_tup[0])
        storage_key = pers_id_tup[2]

        storage = tensor.storage()
        buffer = (ctypes.c_char * storage.nbytes()).from_address(storage.data_ptr())
        data_path = f"archive/data/{storage_key}"
        # print(f"writing tensor for {key!r} to {data_path!r}")
        writer.output.zip_file.writestr(data_path, bytes(buffer))

        reduced_fn, reduced_args = tensor.__reduce_ex__(2)
        assert reduced_fn is torch._utils._rebuild_tensor_v2

        pers_id = _PersId(pers_id_tup)
        reduced_res = (reduced_fn, (pers_id,) + reduced_args[1:])
        return _ReduceRes(reduced_res)

    sds = [i.state_dict() for i in inputs]
    new_sds = state_dicts_merge(sds, outputs, require_all, merge_tensors, merge_others)
    assert len(new_sds) == len(outputs)

    def persistent_id(obj):
        # print("obj", repr(obj).replace("\n", ""))
        # print("actual persistent_id", type(obj), id(obj))
        if isinstance(obj, _PersId):
            return obj.persistent_id

        return None

    for new_sd, output in itertools.zip_longest(new_sds, outputs):
        writer: _OutputWriter = output_writers[id(output)]
        model = { "state_dict": new_sd }

        io_buffer = io.BytesIO()
        pickler = pickle.Pickler(io_buffer, protocol = 2)
        pickler.persistent_id = persistent_id
        pickler.dump(model)
        data = io_buffer.getvalue()
        writer.output.zip_file.writestr("archive/data.pkl", data)


def main(inputs: List[Input], outputs: List[Output], overwrite: bool, half: bool, extended: bool,
         ema_rename_require: bool, ema_rename_optional, ema_strip: bool,
         set_times: bool, use_tmpfile: bool):
    for i in inputs:
        i.open()

    for i in inputs:
        i.model = model = torch_safe_load_dict_lazy(i.zip_file, extended)
        sd = model["state_dict"]

        if ema_rename_require:
            print("replacing model keys with required ema model keys")
            statedict_convert_ema(sd, False, print_stats = True)
        elif ema_rename_optional:
            print("replacing model keys with ema model keys if present")
            statedict_convert_ema(sd, True, print_stats = True)

        if ema_strip:
            print("stripping ema model keys")
            statedict_strip_ema(sd, True)

    # res = inputs_outputs_merge_in_memory(inputs, [i.configs for i in outputs], merge_weighted)
    # print("done")
    # exit()

    for i in outputs:
        output_path = i.path
        if not overwrite and os.path.exists(output_path):
            raise ValueError(f"output_file path exists already, overwriting disabled {output_path!r}")

        if use_tmpfile:
            write_path = f"{output_path}.tmp"
            mode = "w"
            print(f"writing to tmp file {write_path!r}")
        else:
            write_path = output_path
            mode = "w" if overwrite else "x"
            print(f"writing to {output_path!r}, overwrite={overwrite}")

        i.write_path = write_path
        print(f"opening {write_path!r}")
        i.open(mode)
        pass

    inputs_outputs_merge_torch_zip_stream(inputs, outputs, merge_weighted)

    for i in outputs:
        i.close()

        output_path = i.path
        if use_tmpfile:
            if not overwrite and os.path.exists(output_path):
                raise ValueError(f"output_file path exists, didn't before, overwriting disabled {output_path!r}")
            assert i.write_path is not None
            assert i.write_path != output_path
            os.rename(i.write_path, output_path)

        if set_times:
            input_path = inputs[0].path
            cur = os.stat(input_path)
            print(f"setting access/modified times of {input_path!r} on {output_path!r}", (cur.st_atime_ns, cur.st_mtime_ns))
            os.utime(output_path, ns = (cur.st_atime_ns, cur.st_mtime_ns))

    print("done")


if __name__ == "__main__":
    def setup():
        def single(args):
            inputs = [Input(path) for (path, num) in args.input_file]

            factors = [float(num) if "." in num else int(num) for (path, num) in args.input_file]
            outputs = [Output(args.output_file, factors)]
            return inputs, outputs

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--simple", action = "store_true", help = "no BUILD, int keys")
        parser.add_argument("-o", "--overwrite", action = "store_true")
        parser.add_argument("-H", "--half", action = "store_true")

        parser.add_argument("-e", "--ema-rename-try", action = "store_true", help = "if ema keys present replace normal model keys with ema equivalent, ema keys not kept separately")
        parser.add_argument("--ema-rename", action = "store_true", help = "replace normal model keys with ema equivalent, ema keys not kept separately, require ema keys")
        parser.add_argument("-E", "--ema-strip", action = "store_true", help = "strip ema model keys")
        parser.add_argument("-t", "--times", action = "store_true", help = "set same access/modified time on output file as on input file")

        parser.add_argument("-T", "--no-tempfile", action = "store_true", help = "write to output file directly, don't use tempfile and rename")

        sub_parsers = parser.add_subparsers(title = "merge type", required = True)
        single_parser = sub_parsers.add_parser("single")
        single_parser.set_defaults(fn = single)

        single_parser.add_argument("output_file")
        single_parser.add_argument("-i", "--input-file", nargs = 2, action = "append")

        args = parser.parse_args()
        _logging.basicConfig(level = _logging.DEBUG)

        inputs, outputs = args.fn(args)
        main(inputs, outputs, args.overwrite, args.half, not args.simple, args.ema_rename, args.ema_rename_try, args.ema_strip,
             args.times, not args.no_tempfile)


    setup()
