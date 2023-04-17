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

import contextlib
import ctypes
import hashlib
import io
import sys

if sys.version_info[:2] >= (3, 9):
    from typing import Union, Optional, Any, TypeVar
    from collections.abc import Iterable, Callable, Generator, AsyncGenerator, Collection
else:
    from typing import Dict, List, Tuple, Set, Union, Optional, Iterable, Type, Callable, Any, Generator, TypeVar

import torch


# from torch import PyTorchFileWriter


def tensor_serialize_torch(tens: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(tens, buf)
    return buf.getvalue()


def tensor_deserialize_torch(tensor_data: bytes, map_location = None) -> torch.Tensor:
    buf = io.BytesIO(tensor_data)
    tensor = torch.load(buf, map_location = map_location)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected Tensor", tensor)
    return tensor


def tensor_hash_key_torch_pickle(tens: torch.Tensor) -> str:
    # create hash by serializing tensor via torch.save to bytes and hashing that
    hash_key = tensor_serialize_torch(tens)
    return hashlib.sha1(hash_key).hexdigest()


def tensor_hash_key_raw(tens: torch.Tensor) -> str:
    # create hash by hashing raw data buffer and (shape, dtype, requires_grad, stride)
    # directly get buffer from storage data pointer,
    # should be fine on CPU but use at own risk.

    # TODO: check is_contiguous()

    storage = tens.storage()
    if storage.device.type != 'cpu':
        raise ValueError("expected CPU data")
    buffer = (ctypes.c_char * storage.nbytes()).from_address(storage.data_ptr())
    hasher = hashlib.sha1()
    hasher.update(repr((tens.shape, tens.dtype, tens.requires_grad, tens.stride())).encode())
    hasher.update(buffer)
    return hasher.hexdigest()


def repr_hash(thing) -> str:
    rep = repr(thing).encode()
    return hashlib.sha1(rep).hexdigest()


class TensorCache():
    def __init__(
            self,
            path: str,
            tensor_key_fn: Callable[[torch.Tensor], str] = tensor_hash_key_torch_pickle,
            tensor_serialize_fn: Callable[[torch.Tensor], bytes] = tensor_serialize_torch,
            tensor_deserialize_fn: Callable[[bytes], torch.Tensor] = tensor_deserialize_torch,
    ):
        self.tensor_key_fn = tensor_key_fn
        self.tensor_serialize_fn = tensor_serialize_fn
        self.tensor_deserialize_fn = tensor_deserialize_fn

        self.cache = SqliteKeyValueStore(path)

    def get(self, category: Optional[str], key: Union[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if isinstance(key, torch.Tensor):
            key = self.tensor_key_fn(key)

        if not isinstance(key, str):
            raise TypeError("invalid key, expected str", key)

        value = self.cache.getValue(category, key)
        if value is None:
            return None

        return self.tensor_deserialize_fn(value)

    def set(self, category: Optional[str], key: Union[str, torch.Tensor], tensor: torch.Tensor):
        if isinstance(key, torch.Tensor):
            key = self.tensor_key_fn(key)
        if not isinstance(key, str):
            raise TypeError("invalid key, expected str", key)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("invalid tensor, expected Tensor", tensor)

        tensor_bytes = self.tensor_serialize_fn(tensor)
        if not isinstance(tensor_bytes, bytes):
            raise TypeError("invalid tensor_bytes, expected bytes", key)

        self.cache.setValue(category, key, tensor_bytes, commit = True)

    def get_or_create(self, category: Optional[str], key: Union[str, torch.Tensor], creation_fn: Callable[[], torch.Tensor]) -> Optional[torch.Tensor]:
        tensor = self.get(category, key)
        if tensor is not None:
            return tensor

        tensor = creation_fn()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("invalid tensor, expected Tensor", tensor)

        self.set(category, key, tensor)
        return tensor


import re
import threading
import time
import json

import sqlite3


class SqliteKeyValueStore():
    def __init__(self,
                 dbPath: str,
                 checkSameThread: bool = True,
                 connection: Optional[sqlite3.Connection] = None,
                 tablename = "keyvaluecache",
                 ):
        if connection is not None and dbPath:
            raise ValueError("got connection and dbpath", connection, dbPath)

        self.dbPath = dbPath
        if connection is None:
            connection = sqlite3.connect(dbPath, check_same_thread = checkSameThread)
        self.connection = connection
        self.lock = threading.Lock() if not checkSameThread else contextlib.nullcontext()
        self.checkSameThread = checkSameThread

        self.uncommitted = 0
        if not re.match("^\w+$", tablename):
            raise ValueError("invalid table name", tablename)
        self.tablename = tablename

        self.connection.execute(f'''
		CREATE TABLE IF NOT EXISTS {tablename}
			(
			category TEXT NOT NULL,
			key TEXT NOT NULL,
			type TEXT NOT NULL,
			value BLOB NOT NULL,
			storeTime REAL NOT NULL,
			PRIMARY KEY (key, category)
			);
				''')

    def copy(self):
        return SqliteKeyValueStore(self.dbPath, self.checkSameThread)

    def setValue(self, category: str, key: str, value: Any, commit: bool = True, storeTime: Optional[float] = None):
        if not isinstance(category, str):
            raise ValueError("category required", category)

        if isinstance(value, bytes):
            storeValue = value
            valType = "b"
        elif isinstance(value, str):
            storeValue = value
            valType = "s"
        else:
            storeValue = json.dumps(value)
            valType = "j"

        # print("setting", repr(keyStr), type(storeValue), repr(storeValue))
        if storeTime is None:
            storeTime = time.time()
        else:
            storeTime = float(storeTime)

        with self.lock:
            self.connection.execute(f"INSERT OR REPLACE INTO {self.tablename} values (?, ?, ?, ?, ?)", (category, key, valType, storeValue, storeTime))
            if commit:
                self.connection.commit()
                self.uncommitted = 0
            else:
                self.uncommitted += 1

    def commit(self):
        with self.lock:
            self.connection.commit()
            self.uncommitted = 0

    def hasUncommittedChanges(self):
        # return self.connection.in_transaction
        with self.lock:
            return bool(self.uncommitted)

    def getValue(self, category: str, key: str, timestamp: bool = False):
        if not isinstance(category, str):
            raise ValueError("category required", category)

        with self.lock:
            res = self.connection.execute(f"SELECT type, value, storeTime from {self.tablename} where category = ? AND key = ? ", [category, key])
            results = res.fetchall()

        if not results:
            return None

        if len(results) > 1:
            raise ValueError("WTF, multiple results with the same key???", results, category, key)

        item = results[0]
        if len(item) != 3:
            raise ValueError("WTF, invalid result item", item, category, key)

        valType = item[0]
        value = item[1]

        # print('results', type(value), repr(value))
        if not timestamp:
            return self._processValue(valType, value)

        storeTime = float(item[2])
        return storeTime, self._processValue(valType, value)

    def _processValue(self, valType: str, value):
        if valType in "b":
            if not isinstance(value, bytes):
                raise ValueError("expected bytes, got ", type(value), value)
            return value

        if valType in "s":
            if not isinstance(value, str):
                raise ValueError("expected str, got ", type(value), value)
            return value

        elif valType == "j":
            return json.loads(value)
        else:
            raise ValueError("unsupported valtype: ", repr(valType))

    def getKeys(self, category: Optional[str]):
        with self.lock:
            if category is not None:
                res = self.connection.execute(f"SELECT category, key from {self.tablename} where category = ?", [category])
            else:
                res = self.connection.execute(f"SELECT category, key from {self.tablename}")

            if not self.checkSameThread:
                # if possibly threaded then fetch all within lock
                res = list(res)

        for (cat, key) in res:
            yield (cat, key)

    def getKeyCount(self, category: Optional[str]) -> int:
        with self.lock:
            if category is not None:
                res = self.connection.execute(f"SELECT count(key) from {self.tablename} where category = ?", [category])
            else:
                res = self.connection.execute(f"SELECT count(key) from {self.tablename}")

            return int(res.fetchone()[0])

    def getCategories(self):
        with self.lock:
            res = self.connection.execute(f"SELECT distinct category from {self.tablename}")
            if not self.checkSameThread:
                # if possibly threaded then fetch all within lock
                res = list(res)

        for cat in res:
            yield str(cat[0])

    def vacuum(self):
        with self.lock:
            self.connection.execute("vacuum;")

    def close(self):
        with self.lock:
            self.connection.close()
            self.connection = None
