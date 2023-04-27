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
deps: torch pyparsing==3.0.*
optional: tqdm safetensors

usage:
    merging with multiple outputs or complex merge expressions:
        Only loads individual tensors lazily as needed
        so even merging a large number of models into lots of
        configurations should only need a few hundred MBs of RAM.

          ./safe_multi_merge.py \
          multi \
          ./output/folder/target/ \
          -I ./src/models/sd-v1-5.fp16.safe.ckpt SD15 \
          -I ./src/models/sd-v1-4.fp16.safe.ckpt SD14 \
          -I ./src/models/Anything-V3.0.fp16.safe.ckpt Any3 \
          -I ./src/models/StudioGhibli.fp16.safe.ckpt Gibli \
          -I ./src/models/discoElysium.fp16.safe.ckpt DisEl \
          -I ./src/models/DreamBooth1.ckpt DB1 \
          -I ./src/models/DreamBooth2.ckpt DB2 \
          -I ./src/models/DreamBooth3.ckpt DB3 \
          -e ./target/custom_name.ckpt "Any3@0.3 + (Gibli + DisEl@2)@0.5 + DB1@0.2" \  `#(Gibli + DisEl@2) will be weighted merged at 1:2 first then merged with the rest`
          -E "SD15 < ( (DB1-SD15) + (DB2-SD14)@2 )" \     `# will subtract SD15 from DB1, then subtract SD14 from DB2, then weighted merge only the differences at 1:2 ratio, then add that to SD15 ( < means a straight add no merging)`
          -E "Any3 + (SD15 < ((DB1-SD15) + (DB2-SD15) + (DB3-SD14)))"  \  `# get all the differences, weight merge the diffs, add to SD15, weight merge the result with Anything3`
          -E ..... \
          -E ..... \
          -E .....

        # -e "filename.ckpt" "expression"   # -e is with custom name as first argument
        # -E "expression"                   # -E is auto named and saved to output folder target, must be specified with -E.

    simple mode (deprecated), single output file all merged together with weighted averaging with chosen factors:
        (all factors will be normalized to total of 1)

        ./safe_multi_merge.py single output_filepath_name.ckpt \
        -i /model_path/mod1.ckpt 1 \
        -i /model_path/asdf.ckpt 1 \
        -i /model_path/test.ckpt 1 \
        -i /model_path/long_name_whatever_123.ckpt 2

        will merge into one model with weights 0.2, 0.2, 0.2, 0.4 respectively, saved as output_filepath_name.ckpt.

        For auto naming just pass a directory as argument (needs to exist as directory or end with /),
        by default uses model filename unless -I is used and custom name is given:


        ./safe_multi_merge.py single ./target/folder/ \
        -i /model_path/mod1.ckpt 1 \
        -i /model_path/other.ckpt 1.5 \
        -i /model_path/example.ckpt 2.1 \
        -I /model_path/long_name_whatever_123.ckpt Custom_NAME 2

        will be written to: ./target/folder/merged_@UTV_mod1@0.1515_+other@0.2273_+example@0.3182_+Custom_NAME@0.303.ckpt


"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import io
import itertools
import logging
import logging as _logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from zipfile import ZipFile

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from type_utils import ensure_equal, ensure

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar, List, Dict, Tuple
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection, Set
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

from safe_load import pickle_bytes_safe_load_dict, DTYPE_MAP, statedict_convert_ema, statedict_strip_ema, get_archive_name, _build_tensor, _guess_filetype

try:
    import safetensors
except:
    safetensors = None

try:
    import tqdm
except:
    tqdm = None

import torch
import merge_expression
from merge_expression import Add, Mul, Merge, RunExp, Tens, Var, Minus, runexp_to_str

logger = _logging.getLogger(__name__)
IS_DEV = os.environ.get("DEV") == "1"

SKIP_INPUT = object()


@dataclasses.dataclass()
class Input():
    path: str
    ident: Optional[str]
    zip_file: Optional[ZipFile] = None
    model: Optional[dict] = None
    filetype: Optional[str] = None

    def ensure_exists(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

    def open(self):
        if self.zip_file is not None:
            raise ValueError("already open", self.path)

        self.zip_file = ZipFile(self.path, mode = "r")

    def state_dict(self) -> dict:
        return self.model["state_dict"]


@dataclasses.dataclass()
class ExpressionMerge():
    merge: RunExp
    original_expression: Optional[str] = None


Merger = Union[ExpressionMerge, Callable]


@dataclasses.dataclass()
class MergeFn():
    name: str
    matcher: Callable[[str], bool]
    merger: Merger


@dataclasses.dataclass()
class OutputArg():
    path: Optional[str]
    config: Tuple[str, Any]


@dataclasses.dataclass()
class Output():
    path: Optional[str]
    mergers: List[MergeFn]
    missing_key_fallback_fn: Optional[Callable] = None

    write_path: Optional[str] = None
    zip_file: Optional[ZipFile] = None
    zip_file_file: Optional[io.FileIO] = None
    model: Optional[dict] = None

    def open(self, mode: str, compression = zipfile.ZIP_STORED, compresslevel = None,
             with_file = False, torch_writer: bool = False):
        if self.zip_file is not None or self.zip_file_file is not None:
            raise ValueError("already open", self.path, self.zip_file, self.zip_file_file)

        if with_file:
            open_mode = mode.replace("b", "") + "b"
            self.zip_file_file = open_arg = open(self.write_path, open_mode)
        else:
            open_arg = self.write_path

        if torch_writer:
            zf = torch.serialization._open_zipfile_writer(open_arg)

            def writestr(path: str, data: bytes):
                if path == "archive/version":
                    # ignore version, written by the torch writer itself, only needed with normal zipfile
                    # print("ignoring version", data)
                    return
                ensure_equal(path.startswith("archive/"), True, "invalid path", path)  # TODO: may be different in torch 1.13?
                path = path[8:]
                zf.file_like.write_record(path, data, len(data))

            def close():
                pass

            zf.writestr = writestr
            zf.close = close
            self.zip_file = zf
        else:
            self.zip_file = ZipFile(open_arg, mode = mode, compression = compression, compresslevel = compresslevel)

    def open_buffer(self, mode: str, compression = zipfile.ZIP_STORED, compresslevel = None):
        if self.zip_file is not None:
            raise ValueError("already open", self.path)

        buffer = io.BytesIO()
        self.zip_file = ZipFile(buffer, mode = mode, compression = compression, compresslevel = compresslevel)

    def close(self):
        self.zip_file.close()
        self.zip_file = None
        self.zip_file_file = None

    def key_merge(self, key) -> Merger:
        for i in self.mergers:
            if i.matcher(key):
                return i.merger

        raise ValueError("never got merger for key", key)


class LazyTensor():
    def load(self, reload = False, dtype = None):
        raise NotImplementedError()

    def dtype(self):
        raise NotImplementedError()

    def unload(self):
        raise NotImplementedError()

    def load_copy(self):
        raise NotImplementedError()


@dataclasses.dataclass()
class SafetensorsLazyTensor(LazyTensor):
    safe_tensor: Any
    safe_tensor_root: Any
    safe_tensor_key: Any

    def load(self, reload = False, dtype = None):
        pass

    def dtype(self):
        st = self.safe_tensor
        if st is None:
            st = self.safe_tensor_root.get_tensor(self.safe_tensor_key)
        return st.dtype

    def unload(self):
        pass

    def load_copy(self):
        st = self.safe_tensor
        if st is None:
            st = self.safe_tensor_root.get_tensor(self.safe_tensor_key)
        return st.detach().clone()


@dataclasses.dataclass()
class TorchLazyTensor(LazyTensor):
    _ctx: Any
    _zip_file: ZipFile
    _archive_name: str
    _storage_tup: tuple
    _data_path: str

    _storage_offset: int
    _size: torch.Size
    _stride: Union[tuple, int]
    _requires_grad: bool

    tensor: Optional[torch.Tensor] = None

    def load(self, reload = False, dtype = None):
        if self.tensor is not None and not reload:
            return self.tensor

        self.tensor = self.load_copy()
        return self.tensor

    def dtype(self):
        return DTYPE_MAP[self._storage_tup[1]][0]

    def unload(self):
        self.tensor = None

    def load_copy(self):
        return _build_tensor(
            self._zip_file, self._archive_name, self._storage_tup, self._storage_offset, self._size,
            self._stride, self._requires_grad, None,
        )

    def load_meta(self):
        return _build_tensor_meta(self._storage_tup, self._storage_offset, self._size, self._stride)


def _build_tensor_meta(storage, storage_offset, size, stride):
    if storage_offset:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride))

    (storage, dtype_str, index, location, element_count) = storage
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    # return torch.empty(size, stride, dtype = dtype, device = "meta")
    tensor = torch.empty_strided(tuple(size), stride, dtype = dtype, device = "meta")
    return tensor


def _build_lazy_tensor(ctx, zipfile: ZipFile, archive_name: str, storage_tup, storage_offset, size, stride, requires_grad, backward_hooks):
    if storage_offset or backward_hooks:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

    (storage, dtype_str, index, location, element_count) = storage_tup
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    data_path = f"{archive_name}/data/{index}"
    data_size = zipfile.getinfo(data_path).file_size

    expected_size = element_count * dtype_size
    if data_size != expected_size:
        raise ValueError("read unexpected amount of bytes",
                         data_size, expected_size, data_path, element_count, dtype_size)

    return TorchLazyTensor(ctx, zipfile, archive_name, storage_tup, data_path, storage_offset, torch.Size(size), stride, requires_grad)


def torch_safe_load_dict_lazy(model_path_or_zipfile: Union[str, ZipFile], extended: bool = False, tensor_ctx = None):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = ZipFile(model_path_or_zipfile)

    try:
        data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")
        archive_name = "archive"
    except KeyError:
        archive_name = get_archive_name(model_path_or_zipfile, True)
        data_pickle_bytes = model_path_or_zipfile.read(f"{archive_name}/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    build_tensor = functools.partial(_build_lazy_tensor, tensor_ctx, model_path_or_zipfile, archive_name)
    model = pickle_bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
        extended = extended,
    )

    return model


def _merge_use_first(_ctx, vals: dict[str, Any]):
    # print("_merge_use_first!!!", ctx, len(vals), str(vals).replace("\n","")[:100], factors)
    for i in vals.values():
        if i is not None:
            return i
    raise ValueError("no non None value found", vals)


class _MissingTensorError(Exception):
    pass


@dataclasses.dataclass()
class Settings():
    merge_inpainting: bool


@dataclasses.dataclass()
class ParseCtx():
    var_map: Dict
    key: str
    merge_inpainting: bool


class ModelKeyError(KeyError):
    def __init__(self, model_name: str, key_name: str, *args):
        super().__init__(model_name, key_name, *args)
        self.model_name = model_name
        self.key_name = key_name


def _exp_get(ctx: ParseCtx, tens: Tens):
    if isinstance(tens, Merge):
        return _exp_run_merge(ctx, tens)

    if isinstance(tens, Var):
        try:
            return ctx.var_map[tens.name]
        except KeyError:
            raise ModelKeyError(tens.name, ctx.key) from None

    if isinstance(tens, Minus):
        return _exp_run_minus(ctx, tens)

    if isinstance(tens, Add):
        return _exp_run_add(ctx, tens)

    if isinstance(tens, Mul):
        return _exp_run_mul(ctx, tens)

    raise ValueError("unsupported tens", tens)


def _exp_run_minus(ctx: ParseCtx, minus: Minus):
    left = _exp_get(ctx, minus.left)
    right = _exp_get(ctx, minus.right)
    return torch.subtract(left, right)


def _exp_run_add(ctx: ParseCtx, add: Add):
    tensors = [_exp_get(ctx, i) for i in add.tensors]
    res = tensors[0]
    for i in tensors[1:]:
        res = torch.add(res, i)
    return res


def _exp_run_mul(ctx: ParseCtx, mul: Mul):
    item = _exp_get(ctx, mul.item)
    return torch.mul(item, mul.factor)


INPAINT_INPUT_KEY = "model.diffusion_model.input_blocks.0.0.weight"


def _exp_run_merge_inpaint(ctx: ParseCtx, merge: Merge):
    # merging inpainting with non-inpainting, shapes [320, 4, 3, 3] or [320, 9, 3, 3]
    # merges all normal layers according to factors,
    # and all extra inpainting layers according to same factors but full ignoring any without
    # as if they didn't exist.
    # So [Norm, Norm, Inp, Inp] would merge as [0.25,0.25,0.25,0.25] for the first 4 layers
    # and [0.5, 0.5] for the inpainting layers.

    normal_shape = [320, 4, 3, 3]
    inpaint_shape = [320, 9, 3, 3]

    tensors_factors = [(_exp_get(ctx, i.item), i.factor) for i in merge.items]
    for tens, _fact in tensors_factors:
        ensure(list(tens.shape) in (normal_shape, inpaint_shape))

    # merge everything of :4 normal layers as usual
    factor_total = merge.factor_total()
    base = tensors_factors[0][0].clone() * (tensors_factors[0][1] / factor_total)
    for tens, fact in tensors_factors[1:]:
        tens = tens.clone()
        base[:, :4, :, :] += tens[:, :4, :, :] * (fact / factor_total)

    inp_tensors = [(tens, fact) for (tens, fact) in tensors_factors if tens.shape[1] == 9]
    ensure(inp_tensors)

    # merge 5:9 for all that have the inpaint layers
    if len(inp_tensors) == 1:
        # just one inpaint one, keeping the 5:9 from that one as is,
        # overwrite rest with already fully merged base
        inp_tens = inp_tensors[0][0].clone()
        inp_tens[:, :4, :, :] = base[:, :4, :, :]
        return inp_tens

    # multiple with inpaint layers, merge all those
    factor_total = sum(fact for (_t, fact) in inp_tensors)
    inp_base = inp_tensors[0][0].clone() * (inp_tensors[0][1] / factor_total)
    for tens, fact in inp_tensors[1:]:
        inp_base[:, 4:, :, :] += tens[:, 4:, :, :] * (fact / factor_total)

    # copy already merged base to one with 9 inpaint shape
    inp_base[:, :4, :, :] = base[:, :4, :, :]
    return inp_base


def _exp_run_merge(ctx: ParseCtx, merge: Merge):
    factor_total = merge.factor_total()
    fact = merge.items[0].factor / factor_total
    base = _exp_get(ctx, merge.items[0].item) * fact
    for i in merge.items[1:]:
        tens = _exp_get(ctx, i.item)
        fact = i.factor / factor_total
        try:
            base += tens * fact
        except RuntimeError:
            if (
                    ctx.merge_inpainting
                    and ctx.key == INPAINT_INPUT_KEY
                    and base.shape != tens.shape
                    and len(base.shape) == 4
                    and len(tens.shape) == 4
                    and base.shape[0] == tens.shape[0]
                    and base.shape[2:] == tens.shape[2:]
                    and (base.shape[1], tens.shape[1]) in [(4, 9), (9, 4)]
            ):
                return _exp_run_merge_inpaint(ctx, merge)
            raise

    return base


def _exp_run(ctx: ParseCtx, runexp: RunExp):
    if isinstance(runexp.item, Merge):
        return _exp_run_merge(ctx, runexp.item)
    elif isinstance(runexp.item, Add):
        return _exp_run_add(ctx, runexp.item)
    raise ValueError("invalid runexp type", runexp.item)


def merge_tensors(
        conf: Settings, key: str, output: Output,
        tensors: dict[str, torch.Tensor], missing_inputs: List[Input], has_floats: bool, has_non_floats: bool,
) -> torch.Tensor:
    merge = output.key_merge(key)

    try:
        if isinstance(merge, ExpressionMerge):
            pctx = ParseCtx(tensors, key, conf.merge_inpainting)
            return _exp_run(pctx, merge.merge)

        if callable(merge):
            return merge(output, tensors)

    except ModelKeyError as ex:
        logger.warning("model missing required, using fallback choice fn: %s, key: %r, model: %r",
                       output.missing_key_fallback_fn, key, ex.model_name)
        if output.missing_key_fallback_fn is not None:
            return output.missing_key_fallback_fn(output, tensors)
        raise

        # return i.merger(key, ctx, configs, inputs, tensors, missing_inputs, has_floats, has_non_floats)
    raise NotImplementedError("unsupported merge type", merge)


def _precision_arg(state_dicts: List[dict], precision: str = "auto", log = False) -> torch.dtype:
    if precision not in ("fp32", "fp16", "auto"):
        raise ValueError("invalid precision", precision)

    if precision != "auto":
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[precision]

    expected_floats = { torch.float16, torch.float32 }
    # auto precision, check one key shared by all for dtypes,
    # used fp32 if any are fp32 otherwise f16

    shared_keys = set(state_dicts[0].keys())
    for i in state_dicts[1:]:
        shared_keys = shared_keys & i.keys()

    # check the "model." key with the longest name if there is any,
    # otherwise longest key name overall
    model_keys = [i for i in shared_keys if i.startswith("model.")]
    longest_model_key = max(model_keys or shared_keys, key = lambda i: len(i))

    get_dtype = lambda tens_or_lazy: tens_or_lazy.dtype() if isinstance(tens_or_lazy, LazyTensor) else tens_or_lazy.dtype
    dtypes = { get_dtype(sd[longest_model_key]) for sd in state_dicts }
    has_floats = dtypes & expected_floats
    non_floats = dtypes - expected_floats
    if non_floats:
        raise ValueError("got non floats for model keys, huh?", dtypes)

    if len(has_floats) == 1:
        use_precision = list(has_floats)[0]
        if log:
            logger.info("using only seen precision: %s", use_precision)
    else:
        if has_floats != expected_floats:
            raise ValueError("expected fp16 and fp32 here", dtypes)
        use_precision = torch.float32
        if log:
            logger.info("using best seen precision: %s", use_precision)

    return use_precision


def state_dicts_merge(
        input_state_dicts: dict[str, dict],
        merge_contexts: List[Any],
        require_all: bool,
        merge_tensors_fn: Callable[[str, Any, List[Optional[torch.Tensor]], bool, bool], Optional[torch.Tensor]],
        merge_non_tensors_fn: Callable[[str, Any], Any],
        no_mixed: bool = False,
        precision: str = "auto",
        work_iter: Optional[Callable[[Iterable], Iterable]] = None,
) -> List[Dict]:
    state_dicts = list(input_state_dicts.values())
    ensure(state_dicts)

    all_keys = set()
    for sd in state_dicts:
        all_keys = all_keys | sd.keys()

    expected_floats = { torch.float16, torch.float32 }
    use_precision = _precision_arg(state_dicts, precision, True)

    merge_contexts = list(merge_contexts)
    new_sd_tups = [(i, { }) for i in merge_contexts]
    all_keys = sorted(all_keys)
    if work_iter:
        all_keys = work_iter(all_keys)

    for key in all_keys:
        lazy_tensors = { }
        others = { }
        for sid, sd in input_state_dicts.items():
            val = sd.get(key)
            if isinstance(val, LazyTensor):
                lazy_tensors[sid] = val
            else:
                if val is not None:
                    print("ignoring non tensor key", key, type(val))

                lazy_tensors[sid] = None
                others[sid] = val

        if len(others) == len(lazy_tensors):
            for merge_ctx, new_sd in new_sd_tups:
                new_sd[key] = merge_non_tensors_fn(key, others)
            continue

        if require_all and None in lazy_tensors.values():
            missing = { k for (k, v) in lazy_tensors.items() if v is None }
            raise ValueError(f"tensor missing from {len(missing)} state_dicts", key, missing)

        dtypes = { i.dtype() for i in lazy_tensors.values() if i is not None }
        has_floats = dtypes & expected_floats
        non_floats = dtypes - expected_floats

        is_mixed = False
        if has_floats and non_floats:
            # mix of floats and non floats, probably simply halfed
            if no_mixed:
                raise ValueError("dtype mix floats and non floats not allowed", key, dtypes)
            is_mixed = True

        # print(key, lazy_tensors)
        if has_floats:
            tensors = { k: (v.load_copy().to(use_precision) if v is not None else None) for k, v in lazy_tensors.items() }
        else:
            tensors = { k: (v.load_copy() if v is not None else None) for k, v in lazy_tensors.items() }

        # sizes = { i.storage().nbytes() for i in tensors if i is not None }
        # if len(sizes) > 1:
        #     raise ValueError("tensors of various sizes, can't merge", key, sizes, dtypes, is_mixed)

        for merge_ctx, new_sd in new_sd_tups:
            new_sd[key] = merge_tensors_fn(key, tensors, merge_ctx, bool(has_floats), is_mixed)

        del tensors

    return [new_sd for merge_ctx, new_sd in new_sd_tups]


def _tensors_configs_filter(key: str, maybe_tensors: dict[str, Optional[torch.Tensor]]):
    tensors = { }
    missing_input_keys = set()

    for k, v in maybe_tensors.items():
        if v is None:
            missing_input_keys.add(k)
        else:
            tensors[k] = v

    return tensors, missing_input_keys


def _to_inputs(inputs: List[Union[Input, str]]) -> List[Input]:
    use_inputs = []
    for i in inputs:
        if not isinstance(i, Input):
            i = Input(i, i)
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


class _ZeroDimErr(Exception):
    pass


def _all_equal_data(tensors: Iterable[torch.Tensor]):
    for i in tensors:
        if i.dim() == 0:
            raise _ZeroDimErr()

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


def _all_equal(tensors: Iterable[torch.Tensor], key = None):
    try:
        return _all_equal_basic(tensors)
    except _ZeroDimErr:
        # logger.debug("ignoring zero dim _all_equal_data error for %r", key)
        pass

    return False


def _single_tensor_check(tensors: List[torch.Tensor], skip_equal: bool, key = None) -> Optional[torch.Tensor]:
    if len(tensors) == 1:
        return torch.clone(tensors[0])
    elif skip_equal and _all_equal(tensors, key):
        logging.debug("skipping merging %s all equal tensors: %r", len(tensors), key)
        return torch.clone(tensors[0])
    return None


def _unknown_vars(inputs: List[Input], outputs: List[Output]) -> List[str]:
    known_input_ids = { i.ident for i in inputs if i.ident }
    unknown_all = set()
    for out in outputs:
        for merger in out.mergers:
            if isinstance(merger.merger, ExpressionMerge):
                used_vars = merge_expression.runexp_vars(merger.merger.merge)
                unknown = set(used_vars) - known_input_ids
                if unknown:
                    unknown_all.update(unknown)

    return sorted(unknown_all)


def _never_used_inputs(inputs: List[Input], outputs: List[Output]) -> List[int]:
    # returns a list of input position ids which are always skipped by
    # every output config.
    used_input_idxs = set()

    mergers = []
    for out in outputs:
        for merger in out.mergers:
            mergers.append(merger.merger)

    for merge in mergers:
        if isinstance(merge, ExpressionMerge):
            used_vars = merge_expression.runexp_vars(merge.merge)
            for pos, i in enumerate(inputs):
                if i.ident and i.ident in used_vars:
                    used_input_idxs.add(pos)
        elif callable(merge):
            pass
        else:
            raise ValueError("unsupported merge type", merge)

    return [i for i in range(len(inputs)) if i not in used_input_idxs]


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
    ensure_unique(inputs, lambda i: i.ident)

    outputs = [(i if isinstance(i, Output) else Output(None, i)) for i in configs_outputs]
    ensure(outputs, "no outputs given")

    unknown = _unknown_vars(inputs, outputs)
    if unknown:
        raise ValueError("expression used unknown model names",
                         unknown, sorted(i.ident for i in inputs if i.ident))

    unused_input_idxs = _never_used_inputs(inputs, outputs)
    if unused_input_idxs:
        logger.info("not loading %s unused inputs: %s", len(unused_input_idxs),
                    [inputs[idx].ident or f"in[{idx}]" for idx in unused_input_idxs])

    unused_input_idxs_set = set(unused_input_idxs)
    inputs_by_id = { i.ident: i for i in inputs }
    sds = { i.ident: i.state_dict() for pos, i in enumerate(inputs) if pos not in unused_input_idxs_set }

    def merge_tensors(key: str, maybe_tensors: dict[str, Optional[torch.Tensor]], ctx: Output, has_floats: bool, is_mixed: bool):
        ensure_equal(len(sds), len(maybe_tensors))
        tensors, missing_inputs_ids = _tensors_configs_filter(key, maybe_tensors)
        missing_inputs = [inputs_by_id[i] for i in missing_inputs_ids]
        if tensor := _single_tensor_check(list(tensors.values()), skip_equal, key) is not None:
            return tensor
        return merge_fn(key, ctx, tensors, missing_inputs, has_floats, is_mixed)

    new_sds = state_dicts_merge(
        sds, outputs, require_all, merge_tensors, merge_non_tensors,
        precision = precision, work_iter = tqdm.tqdm if tqdm else None,
    )
    return new_sds


try:
    from torch.storage import _TypedStorage as SaveTypedStorage
except:
    from torch.storage import TypedStorage as SaveTypedStorage


class _OutputWriter():
    def __init__(self, output: Output):
        self.output = output
        self.persistent_id_pos = -1
        self.dtype_classes = { }
        for key, val in vars(torch).items():
            if (
                    key and key[0].isupper()
                    and key.endswith("Storage")
                    and isinstance(val, type)
                    and issubclass(val, torch.storage._LegacyStorage)
            ):
                self.dtype_classes[val.dtype] = val

    def persistent_id(self, tensor: torch.Tensor):
        self.persistent_id_pos += 1
        pos = self.persistent_id_pos

        storage = tensor.storage()
        if isinstance(storage, SaveTypedStorage):
            type_str = self.dtype_classes[storage.dtype]
            bytes_len = storage.size()
        else:
            type_str = torch.serialization.normalize_storage_type(type(tensor))
            bytes_len = storage.nbytes()

        return "storage", type_str, str(pos), "cpu", bytes_len


def inputs_outputs_merge_torch_zip_stream(
        inputs: List[Union[Input, str]],
        outputs: List[Output],
        merge_fn: Callable,
        require_all: bool = False,
        precision: str = "auto",
        merge_non_tensors = _merge_use_first,
        skip_equal: bool = True,
):
    ensure_unique(inputs, lambda i: i.ident, ignored_keys = { None })

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
    ensure_unique(inputs, lambda i: i.ident)
    ensure(outputs, "no outputs given")

    output_writers = { id(i): _OutputWriter(i) for i in outputs }

    for ow in output_writers.values():
        ow.output.zip_file.writestr("archive/version", "3\n")

    unused_input_idxs = _never_used_inputs(inputs, outputs)
    if unused_input_idxs:
        logger.info("not loading %s unused inputs: %s", len(unused_input_idxs),
                    [inputs[idx].ident or f"in[{idx}]" for idx in unused_input_idxs])
    unknown = _unknown_vars(inputs, outputs)
    if unknown:
        raise ValueError("expression used unknown model names",
                         unknown, sorted(i.ident for i in inputs if i.ident))

    unused_input_idxs_set = set(unused_input_idxs)
    inputs_by_id = { i.ident: i for i in inputs }
    sds = { i.ident: i.state_dict() for pos, i in enumerate(inputs) if pos not in unused_input_idxs_set }

    def merge_tensors(key: str, maybe_tensors: dict[str, Optional[torch.Tensor]], ctx: Output, has_floats: bool, is_mixed: bool):
        writer: _OutputWriter = output_writers[id(ctx)]

        ensure_equal(len(sds), len(maybe_tensors))
        tensors, missing_inputs_ids = _tensors_configs_filter(key, maybe_tensors)
        missing_inputs = [inputs_by_id[i] for i in missing_inputs_ids]
        tensor = _single_tensor_check(list(tensors.values()), skip_equal, key)
        if tensor is None:
            tensor = merge_fn(key, ctx, tensors, missing_inputs, has_floats, is_mixed)

        # create fake classes that pickle serialize like torch.Tensor
        pers_id_tup = writer.persistent_id(tensor)
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

    new_sds = state_dicts_merge(
        sds, outputs, require_all, merge_tensors, merge_non_tensors,
        precision = precision,
        work_iter = tqdm.tqdm if tqdm else None,
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


def torch_zip_stream(
        output_path: Union[str, Output],
        key_tensor_iter: Iterable[Tuple[str, Union[torch.Tensor, LazyTensor]]],
        open_mode = "x",
        wrap_in_dict: Optional[str] = None,
        with_file = True, torch_writer = True
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

    if isinstance(output_path, Output):
        close = False
        output = output_path
    else:
        close = True
        output = Output(output_path, [], write_path = output_path)
        output.open(open_mode, with_file = with_file, torch_writer = torch_writer)

    ensure(output.zip_file, "output needs to be open")
    writer = _OutputWriter(output)
    writer.output.zip_file.writestr("archive/version", "3\n")

    def _write(tensor: Union[torch.Tensor, LazyTensor]):
        if isinstance(tensor, LazyTensor):
            tensor = tensor.load_copy()

        ensure(isinstance(tensor, torch.Tensor), "expected tensor", type(tensor))

        # create fake classes that pickle serialize like torch.Tensor
        pers_id_tup = writer.persistent_id(tensor)
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

    def persistent_id(obj):
        if isinstance(obj, _PersId):
            return obj.persistent_id

        return None

    sd = { }
    for key, tensor in key_tensor_iter:
        res = _write(tensor)
        sd[key] = res

    if wrap_in_dict:
        model = { wrap_in_dict: sd }
    else:
        model = sd

    io_buffer = io.BytesIO()
    pickler = pickle.Pickler(io_buffer, protocol = 2)
    pickler.persistent_id = persistent_id
    pickler.dump(model)
    data = io_buffer.getvalue()
    writer.output.zip_file.writestr("archive/data.pkl", data)

    if close:
        writer.output.close()


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
        inputs: List[Input], output_arg: OutputArg, output: Output,
        output_dir: Optional[str],
        merge_unet: Optional[bool] = None,
        merge_text_encoder: Optional[bool] = None,
        merge_vae_encoder: Optional[bool] = None,
        add_parent_dirnames: Optional[int] = None,
        check_add_inpaint: bool = False,
        prefix: Optional[str] = None,
        extension: Optional[str] = None,
        original_expression: bool = False,
        relative_factors: bool = True,
):
    add_parent_dirnames = int(add_parent_dirnames or 0)
    if add_parent_dirnames < 0:
        raise ValueError("negative add_parent_dirnames", add_parent_dirnames)

    if output_arg.config[0] == "expression":
        if original_expression:
            full_name = output_arg.config[1].original_expression
        else:
            full_name = runexp_to_str(output_arg.config[1].merge, use_original_factor = not relative_factors)
        # print(full_name)
        pass
    else:
        raise ValueError("unsupported output config", output_arg.config[0])

    full_name = full_name.replace(" ", "_")

    flags = []
    if merge_unet:
        flags.append("U")
    if merge_text_encoder:
        flags.append("T")
    if merge_vae_encoder:
        flags.append("V")

    settings = ""
    if flags:
        settings = "@" + "".join(flags) + "_"

    full_name = (prefix or "") + settings + full_name
    extension = (extension or "").lstrip(".")

    # has_inpaint = False
    if check_add_inpaint:
        unused_input_idxs = set(_never_used_inputs(inputs, [output]))

        reg = re.compile("-inpainting(\.[\w]+)?")
        has_inpaint = any(reg.search(i.path) for pos, i in enumerate(inputs) if pos not in unused_input_idxs)
        if has_inpaint:
            extension = "-inpainting." + extension

    if extension:
        # if not has_inpaint and not extension.startswith("."):
        if not extension.startswith("."):
            extension = "." + extension
        full_name = full_name + extension

    if output_dir:
        return os.path.join(output_dir, full_name)

    # print("full_name", full_name)
    return full_name


def _config_merge_fns(merger, fallback, merge_unet: bool, merge_text_encoder: bool, merge_vae_encoder: bool, ):
    configs = []

    check = []
    if merge_unet:
        check.append(("Unet", "model."))
    if merge_text_encoder:
        check.append(("Text", "cond_stage_model."))
    if merge_vae_encoder:
        check.append(("Vae", "first_stage_model."))

    if check:
        names = ",".join([i[0] for i in check])
        key_starts = [i[1] for i in check]

        def startswith(key: str):
            for start in key_starts:
                if key.startswith(start):
                    return True
            return False

        configs.append(MergeFn(names, startswith, merger))

    configs.append(MergeFn("fallback", lambda key: True, fallback))
    return configs


def _safetensors_load_lazy(path: str, very_lazy: bool) -> dict:
    if safetensors is None:
        raise ValueError("_safetensors_load_lazy but safetensors not installed")

    safe_tens = safetensors.safe_open(path, framework = "pt", device = "cpu")
    sd = { }
    for k in safe_tens.keys():
        if very_lazy:
            sd[k] = SafetensorsLazyTensor(None, safe_tens, k)
        else:
            sd[k] = SafetensorsLazyTensor(safe_tens.get_tensor(k), safe_tens, k)

    return {
        "state_dict": sd
    }


def main(
        inputs: List[Input], output_args: List[OutputArg], output_dir: Optional[str], mode: str,
        overwrite: bool, skip_existing: bool, precision: str, extended: bool, add_parent_dirs: Optional[int],
        name_prefix: Optional[str], extension: Optional[str], inpaint_allowed: bool,
        name_original_expression: bool, name_relative_factors: bool,
        merge_unet: bool, merge_text_encoder: bool, merge_vae_encoder: bool,
        missing_key_fallback: bool,
        ema_rename_require: bool, ema_rename_optional, ema_strip: bool,
        set_times: bool, use_tmpfile: bool):
    if not inputs:
        raise ValueError("no inputs")

    if not merge_unet and not merge_text_encoder and not merge_vae_encoder:
        raise ValueError("disabled merging everything")

    ensure_unique(inputs, lambda i: i.ident, ignored_keys = { None })
    if not output_args:
        raise ValueError("no outputs")

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise ValueError("output dir not found or not a dir", output_dir)

    fallback_fn = _merge_use_first if missing_key_fallback else None
    outputs = []
    for i in output_args:
        configs = _config_merge_fns(i.config[1], _merge_use_first, merge_unet, merge_text_encoder, merge_vae_encoder)
        output = Output(None, configs, fallback_fn)

        if i.path is None:
            if output_dir is None:
                raise ValueError("no name/path given for output and not output_dir, can't create name/path")

            i.path = _make_output_path(
                inputs, i, output, output_dir,
                merge_unet, merge_text_encoder, merge_vae_encoder,
                add_parent_dirs, inpaint_allowed, name_prefix, extension,
                name_original_expression, name_relative_factors,
            )

        if os.path.exists(i.path):
            if os.path.isdir(i.path):
                raise ValueError("expected file path, got dir", i.path)

            if skip_existing:
                print(f"skipping existing output file {str(i.path)!r}")
                continue

            if not overwrite:
                raise ValueError(f"output_file path exists already, overwriting disabled {i.path!r}")

        if i.config[0] == "basic":
            if len(i.config[1].factors) != len(inputs):
                raise ValueError(
                    f"invalid number of output configs, expected {len(inputs)} like inputs but got {len(i.config[1].factors)}",
                    i.config[1]
                )

        output.path = i.path
        outputs.append(output)

    if not output_args:
        raise ValueError("no outputs")

    ensure_unique(outputs, key = lambda out: out.path, ignored_keys = { "/dev/null" })

    unused_inputs_ids = set(id(inputs[i]) for i in _never_used_inputs(inputs, outputs))
    for i in inputs:
        if id(i) in unused_inputs_ids:
            continue
        i.ensure_exists()

    for i in inputs:
        if id(i) in unused_inputs_ids:
            continue

        filetype = _guess_filetype(i.path, "ckpt")
        i.filetype = filetype
        print(f"loading {filetype} from {i.path!r}")
        if filetype == "ckpt":
            i.open()
            i.model = torch_safe_load_dict_lazy(i.zip_file, extended)
        elif filetype == "safetensors":
            i.model = _safetensors_load_lazy(i.path, True)
        else:
            raise ValueError("invalid filetype", filetype)

        sd = i.state_dict()

        if ema_rename_require:
            print("replacing model keys with required ema model keys")
            statedict_convert_ema(sd, False, print_stats = True)
        elif ema_rename_optional:
            print("replacing model keys with ema model keys if present")
            statedict_convert_ema(sd, True, print_stats = True)

        if ema_strip:
            print("stripping ema model keys")
            statedict_strip_ema(sd, True)

    settings = Settings(inpaint_allowed)
    merge_fn = functools.partial(merge_tensors, settings)
    # configs = [_config_merge_fns(i.config[1], _merge_use_first, merge_unet, merge_text_encoder, merge_vae_encoder) for i in output_args]
    # res = inputs_outputs_merge_in_memory(inputs, configs, merge_fn, precision = precision)
    # print(len(res))
    # exit()

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
        print(f"opening merge output file {write_path!r}, merge: {[(a.name, a.merger) for a in i.mergers]}")
        i.open(mode, with_file = True, torch_writer = True)

    start = time.monotonic()
    inputs_outputs_merge_torch_zip_stream(inputs, outputs, merge_fn, precision = precision)
    end = time.monotonic()
    diff = end - start
    for i in outputs:
        i.close()

        output_path = i.path
        if use_tmpfile:
            if os.path.exists(output_path):
                if os.path.isdir(output_path):
                    raise ValueError("expected file path, got dir", output_path)
                if not overwrite:
                    raise ValueError(f"output_file path exists already, overwriting disabled {output_path!r}")

            assert i.write_path is not None
            assert i.write_path != output_path
            os.rename(i.write_path, output_path)

        print(f"saved {i.path!r}")

        if set_times:
            input_path = inputs[0].path
            cur = os.stat(input_path)
            print(f"setting access/modified times of {input_path!r} on {output_path!r}", (cur.st_atime_ns, cur.st_mtime_ns))
            os.utime(output_path, ns = (cur.st_atime_ns, cur.st_mtime_ns))

    print(f"took {diff:.2f} secs ({diff / 60:.2f} mins)")
    print("done")


if __name__ == "__main__":
    def setup():
        parse_num = lambda num: float(num) if "." in num else int(num)
        parsers = []

        def parse_expression(expression: str) -> ExpressionMerge:
            try:
                if not parsers:
                    p = merge_expression.make_parser(True)
                    parsers.append(p)
                else:
                    p = parsers[0]

                merge = merge_expression.parse_run_expression(p, expression)
                return ExpressionMerge(merge, expression)
            except Exception as ex:
                raise argparse.ArgumentTypeError(f"invalid expression {expression!r}: {ex}")

        def multi(args):
            inputs_basic = [Input(path, None) for path in args.input_file or []]
            inputs_named = [Input(path, name) for (path, name) in args.input_file_named or []]
            inputs = inputs_basic + inputs_named

            exp_unnamed = [
                OutputArg(None, ("expression", parse_expression(exp))) for exp in args.output_expression or []
            ]
            exp_named = [
                OutputArg(path, ("expression", parse_expression(exp))) for (path, exp) in args.output_expression_file or []
            ]
            outputs = exp_unnamed + exp_named

            return inputs, outputs, args.output_dir

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--simple", action = "store_true", help = "no BUILD, int keys")
        parser.add_argument("-o", "--overwrite", action = "store_true")
        parser.add_argument("-S", "--skip-existing", action = "store_true")
        parser.add_argument("-v", "--verbose", action = "store_true")

        parser.add_argument("-T", "--no-merge-text-encoder", action = "store_true")
        parser.add_argument("-V", "--no-merge-vae", action = "store_true")
        parser.add_argument("-U", "--no-merge-unet", action = "store_true")

        parser.add_argument("-I", "--no-inpaint", action = "store_true")

        parser.add_argument("-p", "--precision", choices = ["auto", "fp16", "fp32"], default = "auto",
                            help = '"auto" uses "fp32" if any of the tensors are fp32 otherwise uses "fp16"')

        parser.add_argument("-e", "--ema-rename-try", action = "store_true",
                            help = "if ema keys present replace normal model keys with ema equivalent, ema keys not kept separately")
        parser.add_argument("--ema-rename", action = "store_true",
                            help = "replace normal model keys with ema equivalent, ema keys not kept separately, require ema keys")
        parser.add_argument("-E", "--ema-strip", action = "store_true",
                            help = "strip ema model keys")
        parser.add_argument("-t", "--times", action = "store_true",
                            help = "set same access/modified time on output file as on input file")

        parser.add_argument("-N", "--no-tempfile", action = "store_true", help = "write to output file directly, don't use tempfile and rename")
        parser.add_argument("--name-ext", default = "ckpt")
        parser.add_argument("--name-prefix", default = "merged_")
        parser.add_argument("-f", "--name-original-factors", action = "store_true")
        parser.add_argument("--name-original-expression", action = "store_true")
        parser.add_argument("-m", "--missing-key-first-fallback", action = "store_true",
                            help = "if a needed key is missing in one of the models fall back to the first available one")

        parser.add_argument("-P", "--parent-dirs", type = int,
                            help = "add the names of up to [n] parent directories in front of each input name when generating output filename")

        sub_parsers = parser.add_subparsers(title = "merge type", required = True)

        multi_parser = sub_parsers.add_parser("multi")
        multi_parser.set_defaults(fn = multi, mode = "multi")
        multi_parser.add_argument("-i", "--input-file", action = "append")
        multi_parser.add_argument("-I", "--input-file-named", nargs = 2, action = "append")

        multi_parser.add_argument("-o", "--output-file", nargs = 2, action = "append")
        multi_parser.add_argument("-O", "--output", action = "append",
                                  help = "auto named output file, output-dir required")

        multi_parser.add_argument("-e", "--output-expression-file", nargs = 2, action = "append")
        multi_parser.add_argument("-E", "--output-expression", action = "append",
                                  help = "auto named output file, output-dir required")
        # multi_parser.add_argument("-l", "--no-lazyload-inputs", action = "store_true")

        multi_parser.add_argument("output_dir", nargs = "?")

        args = parser.parse_args()
        _logging.basicConfig(level = _logging.DEBUG if args.verbose else _logging.INFO)

        inputs, outputs, output_dir = args.fn(args)
        # print(args)
        # print(inputs)
        # print(outputs)
        # exit()
        main(
            inputs, outputs, output_dir, args.mode,
            args.overwrite, args.skip_existing, args.precision, not args.simple, args.parent_dirs,
            args.name_prefix, args.name_ext, not args.no_inpaint,
            args.name_original_expression, not args.name_original_factors,
            not args.no_merge_unet, not args.no_merge_text_encoder, not args.no_merge_vae,
            args.missing_key_first_fallback,
            args.ema_rename, args.ema_rename_try, args.ema_strip,
            args.times, not args.no_tempfile,
        )


    setup()
