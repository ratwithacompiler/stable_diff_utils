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
import logging
import logging as _logging
import os
import sys
import zipfile
from pathlib import Path
from zipfile import ZipFile

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Dict
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection, Set
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

from safe_load import pickle_bytes_safe_load_dict, DTYPE_MAP, statedict_convert_ema, statedict_strip_ema, _build_tensor

import torch

logger = _logging.getLogger(__name__)
IS_DEV = os.environ.get("DEV") == "1"

SKIP_INPUT = object()


@dataclasses.dataclass()
class Input():
    path: str
    ident: Optional[str]
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

    def dtype(self):
        return DTYPE_MAP[self._storage_tup[1]][0]

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


def merge_weighted(_ctx, tensors: List[torch.Tensor], factors: List[Union[int, float]]) -> torch.Tensor:
    assert len(tensors) == len(factors)
    total = sum(factors)
    factors = [i / total for i in factors]

    base: torch.Tensor = tensors[0] * factors[0]
    for pos, t in enumerate(tensors):
        if not pos:
            continue
        base = base + (tensors[pos] * factors[pos])
    return base


def _merge_use_first(ctx, vals: List[Any], factors):
    # print("_merge_use_first!!!", ctx, len(vals), str(vals).replace("\n","")[:100], factors)
    return vals[0]


def merge_tensors(
        model_merge_fn: Callable,
        text_merge_fn: Callable,
        fallback_merge_fn: Callable,
        key: str,
        configs: List,
        tensors: List[torch.Tensor],
        missing_inputs: List[Input],
        has_floats: bool, has_non_floats: bool,
) -> torch.Tensor:
    if key.startswith("model."):
        if has_non_floats or not has_floats:
            raise ValueError("invalid model tensors", has_floats, has_non_floats, tensors)
        if missing_inputs:
            raise ValueError("missing model tensors", missing_inputs)
        return model_merge_fn(None, tensors, configs)
    elif key.startswith("cond_stage_model."):
        if has_non_floats or not has_floats:
            raise ValueError("invalid text tensors", has_floats, has_non_floats, tensors)
        if missing_inputs:
            raise ValueError("missing text tensors", missing_inputs)
        return text_merge_fn(None, tensors, configs)

    return fallback_merge_fn(None, tensors, configs)


def state_dicts_merge(
        state_dicts: List[dict],
        merge_contexts: List[Any],
        require_all: bool,
        merge_tensors_fn: Callable[[str, Any, List[Optional[torch.Tensor]], bool, bool], Optional[torch.Tensor]],
        merge_non_tensors_fn: Callable[[str, Any, List[Optional[Any]]], Any],
        no_mixed: bool = False,
        precision: str = "auto",
        never_load_inputs: Optional[Set[int]] = None,
) -> List[Dict]:
    if precision not in ("fp32", "fp16", "auto"):
        raise ValueError("invalid precision", precision)

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

    if precision == "auto":
        use_precision = None
    else:
        use_precision = {
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[precision]

    expected_floats = { torch.float16, torch.float32 }
    merge_contexts = list(merge_contexts)
    new_sd_tups = [(i, { }) for i in merge_contexts]
    all_keys = sorted(all_keys)
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
                new_sd[key] = merge_non_tensors_fn(key, merge_ctx, others)
            continue

        if require_all and None in lazy_tensors:
            missing = [pos for (pos, i) in enumerate(lazy_tensors) if i is None]
            raise ValueError(f"tensor missing from {len(missing)} state_dicts", key, missing)

        for pos in never_load_inputs or []:
            lazy_tensors[pos] = None

        dtypes = { i.dtype() for i in lazy_tensors if i is not None }
        has_floats = dtypes & expected_floats
        non_floats = dtypes - expected_floats

        is_mixed = False
        if has_floats and non_floats:
            # mix of floats and non floats, probably simply halfed
            if no_mixed:
                raise ValueError("dtype mix floats and non floats not allowed", key, dtypes)
            is_mixed = True

        # print(key, lazy_tensors)
        tensors = []
        for pos, i in enumerate(lazy_tensors):
            tensors.append(i.load_copy() if i is not None else None)

        if use_precision is None and has_floats:
            # auto precision mode
            if len(has_floats) == 1:
                use_precision = list(has_floats)[0]
                logger.debug("using only seen precision: %s", use_precision)
            else:
                if has_floats != expected_floats:
                    raise ValueError("expected fp16 and fp32 here")
                use_precision = torch.float32
                logger.debug("using best seen precision: %s", use_precision)

        if has_floats:
            tensors = [i.to(use_precision) if i is not None else None for i in tensors]

        sizes = { i.storage().nbytes() for i in tensors if i is not None }
        if len(sizes) > 1:
            raise ValueError("tensors of various sizes, can't merge", key, sizes)

        for merge_ctx, new_sd in new_sd_tups:
            new_sd[key] = merge_tensors_fn(key, merge_ctx, tensors, bool(has_floats), is_mixed)

        del tensors

    return [new_sd for merge_ctx, new_sd in new_sd_tups]


def _tensors_configs_filter(key: str, ctx: Output, inputs: List[Input], maybe_tensors: List[Optional[torch.Tensor]]):
    # filter out None tensors and return with list of corresponding configs
    assert len(maybe_tensors) == len(inputs)
    assert len(maybe_tensors) == len(ctx.configs)

    combined = list(itertools.zip_longest(maybe_tensors, ctx.configs, inputs))
    # filter out ones without tensor of that key in input or with config set to skip
    with_tensor = [(t, c) for (t, c, i) in combined if t is not None and c is not SKIP_INPUT]
    missing_key = [i for (t, c, i) in combined if t is None and c is not SKIP_INPUT]

    if not with_tensor:
        raise ValueError("no tensors", maybe_tensors, with_tensor)

    tensors = [t for (t, c) in with_tensor]
    configs = [c for (t, c) in with_tensor]
    return tensors, configs, missing_key


def _to_inputs(inputs: List[Union[Input, str]]) -> List[Input]:
    use_inputs = []
    for i in inputs:
        if not isinstance(i, Input):
            i = Input(i, None)
            i.open()
        use_inputs.append(i)
    return use_inputs


def _all_equal_basic(tensors: Iterable[torch.Tensor]):
    tensors = list(tensors)
    if not tensors:
        raise ValueError("no tensors")

    first = tensors[0]
    for i in tensors[1:]:
        if not torch.equal(i, first):
            return False

    return True


def _all_equal_data(tensors: Iterable[torch.Tensor]):
    tensors = [i.view(dtype = torch.uint8) for i in tensors]
    if not tensors:
        raise ValueError("no tensors")

    first = tensors[0]
    for i in tensors[1:]:
        if not torch.equal(i, first):
            return False

    return True


def _all_equal_diff(tensors: Iterable[torch.Tensor]):
    tensors = [i.view(dtype = torch.uint8) for i in tensors]
    if not tensors:
        raise ValueError("no tensors")

    first = tensors[0]
    diffs = []
    for i in tensors[1:]:
        # diff = torch.bitwise_xor(i, first)
        diff = torch.sub(i, first)
        diff = torch.sum(diff)
        diffs.append(diff.item())

    return sum(diffs), diffs


def _all_equal(tensors: Iterable[torch.Tensor]):
    e1 = _all_equal_basic(tensors)
    e2 = _all_equal_data(tensors)
    if e1 != e2:
        raise ValueError("wtf??", e1, e2)
    return e1


def inputs_outputs_merge_in_memory(
        inputs: List[Union[Input, str]],
        configs_outputs: List[Union[Any, Output]],
        merge_fn: Callable,
        require_all: bool = False,
        precision: str = "auto",
        merge_non_tensors = _merge_use_first,
        skip_equal: bool = True,
):
    inputs = _to_inputs(inputs)

    def merge_tensors(key: str, ctx: Output, maybe_tensors: List[Optional[torch.Tensor]], has_floats: bool, is_mixed: bool):
        tensors, configs, missing_inputs = _tensors_configs_filter(key, ctx, inputs, maybe_tensors)
        if len(tensors) == 1:
            return torch.clone(tensors[0])
        elif skip_equal and _all_equal(tensors):
            logging.debug("skipping merging %s all equal tensors: %r", len(tensors), key)
            return torch.clone(tensors[0])
        return merge_fn(key, configs, tensors, missing_inputs, has_floats, is_mixed)

    sds = [i.state_dict() for i in inputs]
    outputs = [(i if isinstance(i, Output) else Output(None, i)) for i in configs_outputs]
    unused_inputs = _never_used_inputs(outputs)
    new_sds = state_dicts_merge(
        sds, outputs, require_all, merge_tensors, merge_non_tensors,
        precision = precision, never_load_inputs = unused_inputs,
    )
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


def _never_used_inputs(outputs: List[Output]) -> List[int]:
    # returns a list of input position ids which are always skipped by
    # every output config.

    input_cnts = { len(i.configs) for i in outputs }
    if len(input_cnts) != 1:
        raise ValueError("different input counts", input_cnts)
    input_cnt = list(input_cnts)[0]

    use = [SKIP_INPUT for _ in range(input_cnt)]
    for out in outputs:
        for pos, conf in enumerate(out.configs):
            if conf is not SKIP_INPUT:
                use[pos] = True

    never_used_ids = [pos for (pos, i) in enumerate(use) if i is SKIP_INPUT]
    return never_used_ids


def inputs_outputs_merge_torch_zip_stream(
        inputs: List[Union[Input, str]],
        outputs: List[Output],
        merge_fn: Callable,
        require_all: bool = False,
        precision: str = "auto",
        merge_non_tensors = _merge_use_first,
        skip_equal: bool = True,
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

    def merge_tensors(key: str, ctx: Output, maybe_tensors: List[Optional[torch.Tensor]], has_floats: bool, is_mixed: bool):
        writer: _OutputWriter = output_writers[id(ctx)]

        tensors, configs, missing_inputs = _tensors_configs_filter(key, ctx, inputs, maybe_tensors)
        if len(tensors) == 1:
            tensor = torch.clone(tensors[0])
        elif skip_equal and _all_equal(tensors):
            logging.debug("skipping merging %s all equal tensors: %r", len(tensors), key)
            tensor = torch.clone(tensors[0])
        else:
            # print(f"diff {key!r:<100}", _all_equal_diff(tensors))
            tensor = merge_fn(key, configs, tensors, missing_inputs, has_floats, is_mixed)

        # create fake classes that pickle serialize like torch.Tensor
        pers_id_tup = writer.persistent_id(tensor.storage())
        if pers_id_tup[0] != "storage":
            raise ValueError("expected storage as persistent id 0", pers_id_tup[0])
        storage_key = pers_id_tup[2]

        storage = tensor.storage()
        buffer = (ctypes.c_char * storage.nbytes()).from_address(storage.data_ptr())
        data_path = f"archive/data/{storage_key}"
        writer.output.zip_file.writestr(data_path, bytes(buffer))

        reduced_fn, reduced_args = tensor.__reduce_ex__(2)
        assert reduced_fn is torch._utils._rebuild_tensor_v2

        pers_id = _PersId(pers_id_tup)
        reduced_res = (reduced_fn, (pers_id,) + reduced_args[1:])
        return _ReduceRes(reduced_res)

    sds = [i.state_dict() for i in inputs]
    unused_inputs = _never_used_inputs(outputs)
    new_sds = state_dicts_merge(
        sds, outputs, require_all, merge_tensors, merge_non_tensors,
        precision = precision, never_load_inputs = unused_inputs,
    )
    assert len(new_sds) == len(outputs)

    def persistent_id(obj):
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


def ensure_unique(items, key = None, ignored_keys: Optional[Set] = None):
    by_key = { }
    for i in items:
        ik = key(i) if key is not None else i
        if ignored_keys is not None and ik in ignored_keys:
            continue

        by_key.setdefault(ik, []).append(i)

    by_key = { k: (len(v), v) for k, v in by_key.items() if len(v) > 1 }
    if by_key:
        raise ValueError("multiple items with same key", by_key)


def _cleaned_name(name: str):
    parts = name.split(".")
    while parts and parts[-1] in ("ckpt", "fp16", "fp32", "safe"):
        parts.pop()
    return ".".join(parts)


def _make_output_path(
        inputs: List[Input], output: Output, output_dir: Optional[str],
        add_parent_dirnames: Optional[int] = None,
        prefix: Optional[str] = None,
        extension: Optional[str] = None,
):
    add_parent_dirnames = int(add_parent_dirnames or 0)
    if add_parent_dirnames < 0:
        raise ValueError("negative add_parent_dirnames", add_parent_dirnames)

    parts = []

    for i, factor in itertools.zip_longest(inputs, output.configs):
        if factor is SKIP_INPUT:
            continue

        i: Input

        if i.ident:
            name = i.ident
        else:
            path = Path(i.path)
            name = _cleaned_name(path.name)
            if add_parent_dirnames:
                adding = [i.name for i in path.parents][-add_parent_dirnames:]
                name = "_".join(adding + [name])

        p = f"{name}@{factor}"
        parts.append(p)

    full_name = "_+_".join(parts)
    full_name = (prefix or "") + full_name
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        full_name = full_name + extension

    if output_dir:
        return os.path.join(output_dir, full_name)
    return full_name


def main(
        inputs: List[Input], outputs: List[Output], output_dir: Optional[str],
        overwrite: bool, precision: str, extended: bool, add_parent_dirs: Optional[int],
        name_prefix: Optional[str], extension: Optional[str],
        ema_rename_require: bool, ema_rename_optional, ema_strip: bool,
        set_times: bool, use_tmpfile: bool):
    if not inputs:
        raise ValueError("no inputs")

    ensure_unique(inputs, lambda i: i.ident, ignored_keys = { None })
    if not outputs:
        raise ValueError("no outputs")

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise ValueError("output dir not found or not a dir", output_dir)

    merge_tensors_fn = functools.partial(merge_tensors, merge_weighted, _merge_use_first, _merge_use_first)
    # res = inputs_outputs_merge_in_memory(inputs, [i.configs for i in outputs], merge_tensors_fn, precision = precision)

    for i in outputs:
        # Check for already existing errors in its own first pass
        # to not create unnecessary empty output files in case of any errors.
        output_path = i.path
        if not overwrite and os.path.exists(output_path):
            raise ValueError(f"output_file path exists already, overwriting disabled {output_path!r}")

        if len(i.configs) != len(inputs):
            raise ValueError(f"invalid number of output configs, expected {len(inputs)} like inputs but got {len(i.configs)}")

        if i.path is None:
            if output_dir is None:
                raise ValueError("no name/path given for output and not output_dir, can't create name/path")

            i.path = _make_output_path(inputs, i, output_dir, add_parent_dirs, name_prefix, extension)

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

    ensure_unique(outputs, key = lambda out: out.path, ignored_keys = { "/dev/null" })

    for i in outputs:
        output_path = i.path
        if use_tmpfile:
            write_path = f"{output_path}.tmp"
            mode = "w"
            print(f"writing to tmp file {write_path!r}")
        else:
            write_path = output_path
            mode = "w" if overwrite else "x"
            print(f"writing to {output_path!r}, overwrite={overwrite}")

        i.write_path = write_path
        print(f"opening merge output file {write_path!r}, factors: {i.configs}")
        i.open(mode)

    inputs_outputs_merge_torch_zip_stream(inputs, outputs, merge_tensors_fn, precision = precision)

    for i in outputs:
        i.close()

        output_path = i.path
        if use_tmpfile:
            if not overwrite and os.path.exists(output_path):
                raise ValueError(f"output_file path exists, didn't before, overwriting disabled {output_path!r}")
            assert i.write_path is not None
            assert i.write_path != output_path
            os.rename(i.write_path, output_path)

        print(f"saved {i.path!r}")

        if set_times:
            input_path = inputs[0].path
            cur = os.stat(input_path)
            print(f"setting access/modified times of {input_path!r} on {output_path!r}", (cur.st_atime_ns, cur.st_mtime_ns))
            os.utime(output_path, ns = (cur.st_atime_ns, cur.st_mtime_ns))

    print("done")


if __name__ == "__main__":
    def setup():
        parse_num = lambda num: float(num) if "." in num else int(num)

        def single(args):
            tups_unnamed = [(parse_num(num), Input(path, None)) for (path, num) in args.input_file or []]
            tups_named = [(parse_num(num), Input(path, name)) for (path, name, num) in args.input_file_named or []]
            tups_all = tups_unnamed + tups_named

            factors = [i[0] for i in tups_all]
            inputs = [i[1] for i in tups_all]

            # if os.path.isdir(args.output_file_or_dir):
            if args.output_file_or_dir and args.output_file_or_dir[-1] in ("/", "\\"):
                # is a dir, auto named later into output_dir
                output_dir = args.output_file_or_dir
                outputs = [Output(None, factors)]
            else:
                output_dir = None
                outputs = [Output(args.output_file_or_dir, factors)]

            return inputs, outputs, output_dir

        def parse_factors(factors_arg: str):
            factors = factors_arg.split(",")
            factors = [(SKIP_INPUT if i.strip().lower() in ("s", "skip") else parse_num(i)) for i in factors]
            return factors

        def multi(args):
            inputs_basic = [Input(path, None) for path in args.input_file or []]
            inputs_named = [Input(path, name) for (path, name) in args.input_file_named or []]
            inputs = inputs_basic + inputs_named

            outputs_unnamed = [Output(None, parse_factors(factors)) for factors in args.output or []]
            outputs_named = [Output(path, parse_factors(factors)) for (path, factors) in args.output_file or []]
            outputs = outputs_unnamed + outputs_named

            return inputs, outputs, args.output_dir

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--simple", action = "store_true", help = "no BUILD, int keys")
        parser.add_argument("-o", "--overwrite", action = "store_true")

        parser.add_argument("-p", "--precision", choices = ["auto", "fp16", "fp32"], default = "auto",
                            help = '"auto" uses "fp32" if any of the tensors are fp32 otherwise uses "fp16"')

        parser.add_argument("-e", "--ema-rename-try", action = "store_true", help = "if ema keys present replace normal model keys with ema equivalent, ema keys not kept separately")
        parser.add_argument("--ema-rename", action = "store_true", help = "replace normal model keys with ema equivalent, ema keys not kept separately, require ema keys")
        parser.add_argument("-E", "--ema-strip", action = "store_true", help = "strip ema model keys")
        parser.add_argument("-t", "--times", action = "store_true", help = "set same access/modified time on output file as on input file")

        parser.add_argument("-T", "--no-tempfile", action = "store_true", help = "write to output file directly, don't use tempfile and rename")
        parser.add_argument("--name-ext", default = "ckpt")
        parser.add_argument("--name-prefix", default = "merged_")

        parser.add_argument("-n", "--add-parent-dirs", type = int,
                            help = "add the names of up to [n] parent directories in front of each input name when generating output filename")

        sub_parsers = parser.add_subparsers(title = "merge type", required = True)
        single_parser = sub_parsers.add_parser("single")
        single_parser.set_defaults(fn = single)

        single_parser.add_argument("-i", "--input-file", nargs = 2, action = "append")
        single_parser.add_argument("-I", "--input-file-named", nargs = 3, action = "append")
        single_parser.add_argument("output_file_or_dir")

        multi_parser = sub_parsers.add_parser("multi")
        multi_parser.set_defaults(fn = multi)
        multi_parser.add_argument("-i", "--input-file", action = "append")
        multi_parser.add_argument("-I", "--input-file-named", nargs = 2, action = "append")

        multi_parser.add_argument("-o", "--output-file", nargs = 2, action = "append")
        multi_parser.add_argument("-O", "--output", action = "append",
                                  help = "auto named output file, output-dir required")

        multi_parser.add_argument("output_dir", nargs = "?")

        args = parser.parse_args()
        _logging.basicConfig(level = _logging.DEBUG)

        inputs, outputs, output_dir = args.fn(args)
        # print(args)
        # print(inputs)
        # print(outputs)
        # exit()
        main(
            inputs, outputs, output_dir,
            args.overwrite, args.precision, not args.simple, args.add_parent_dirs,
            args.name_prefix, args.name_ext,
            args.ema_rename, args.ema_rename_try, args.ema_strip,
            args.times, not args.no_tempfile,
        )


    setup()
