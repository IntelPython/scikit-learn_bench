# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from ..utils.bench_case import get_bench_case_value
from ..utils.logger import logger


def convert_data(data, dformat: str, order: str, dtype: str, device: str = None):
    if isinstance(data, csr_matrix) and dformat != "csr_matrix":
        data = data.toarray()
    if dtype == "preserve":
        dtype = None
    if order == "F":
        data = np.asfortranarray(data, dtype=dtype)
    elif order == "C":
        data = np.ascontiguousarray(data, dtype=dtype)
    else:
        raise ValueError(f"Unknown data order {order}")
    if dformat == "numpy":
        return data
    elif dformat == "pandas":
        if data.ndim == 1:
            return pd.Series(data)
        return pd.DataFrame(data)
    elif dformat == "dpnp":
        import dpnp

        return dpnp.array(data, dtype=dtype, order=order, device=device)
    elif dformat == "dpctl":
        import dpctl.tensor

        return dpctl.tensor.asarray(data, dtype=dtype, order=order, device=device)
    elif dformat.startswith("modin"):
        if dformat.endswith("ray"):
            os.environ["MODIN_ENGINE"] = "ray"
        elif dformat.endswith("dask"):
            os.environ["MODIN_ENGINE"] = "dask"
        elif dformat.endswith("unidist"):
            os.environ["MODIN_ENGINE"] = "unidist"
            os.environ["UNIDIST_BACKEND"] = "mpi"
        else:
            logger.info(
                "Modin engine is unknown or not specified. Default engine will be used."
            )

        import modin.pandas as modin_pd

        if data.ndim == 1:
            return modin_pd.Series(data)
        return modin_pd.DataFrame(data)
    elif dformat == "cudf":
        import cudf

        if data.ndim == 1:
            return cudf.Series(data)
        if order == "C":
            logger.warning("cudf.DataFrame is not compatible with C data order")
        return cudf.DataFrame(data)
    elif dformat == "cupy":
        import cupy

        return cupy.array(data)
    else:
        raise ValueError(f"Unknown data format {dformat}")


def train_test_split_wrapper(*args, **kwargs):
    if "ignore" in kwargs:
        result = []
        for arg in args:
            result += [arg, arg]
        return result
    else:
        return train_test_split(*args, **kwargs)


def split_and_transform_data(bench_case, data, data_description):
    if "default_split" in data_description:
        split_kwargs = data_description["default_split"].copy()
    else:
        split_kwargs = {"random_state": 42}
    split_kwargs.update(get_bench_case_value(bench_case, "data:split_kwargs", dict()))
    x = data["x"]
    if "y" in data:
        y = data["y"]
        x_train, x_test, y_train, y_test = train_test_split_wrapper(x, y, **split_kwargs)
    else:
        x_train, x_test = train_test_split_wrapper(x, **split_kwargs)
        y_train, y_test = None, None

    distributed_split = get_bench_case_value(bench_case, "data:distributed_split", None)
    if distributed_split == "rank_based":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        n_train = len(x_train)
        n_test = len(x_test)

        train_start = rank * n_train // size
        train_end = (1 + rank) * n_train // size
        test_start = rank * n_test // size
        test_end = (1 + rank) * n_test // size

        if "y" in data:
            x_train, y_train = (
                x_train[train_start:train_end],
                y_train[train_start:train_end],
            )
            x_test, y_test = x_test[test_start:test_end], y_test[test_start:test_end]
        else:
            x_train = x_train[train_start:train_end]
            x_test = x_test[test_start:test_end]

    device = get_bench_case_value(bench_case, "algorithm:device", None)
    common_data_format = get_bench_case_value(bench_case, "data:format", "pandas")
    common_data_order = get_bench_case_value(bench_case, "data:order", "F")
    common_data_dtype = get_bench_case_value(bench_case, "data:dtype", "float64")

    data_dict = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    if "n_classes" in data_description:
        required_label_dtype = "int"
    else:
        required_label_dtype = None

    for subset_name, subset_content in data_dict.items():
        if subset_content is None:
            continue
        is_label = subset_name.startswith("y")

        data_format = get_bench_case_value(
            bench_case, f"data:{subset_name}:format", common_data_format
        )
        data_order = get_bench_case_value(
            bench_case, f"data:{subset_name}:order", common_data_order
        )
        data_dtype = get_bench_case_value(
            bench_case, f"data:{subset_name}:dtype", common_data_dtype
        )

        if is_label and required_label_dtype is not None:
            data_dtype = required_label_dtype

        converted_data = convert_data(
            subset_content, data_format, data_order, data_dtype, device
        )
        data_dict[subset_name] = converted_data
        if not is_label:
            data_description[subset_name] = {
                "format": data_format,
                "order": data_order,
                "dtype": data_dtype,
                "samples": converted_data.shape[0],
            }
            if len(converted_data.shape) == 2 and converted_data.shape[1] > 1:
                data_description[subset_name]["features"] = converted_data.shape[1]

    return (
        (data_dict[name] for name in ["x_train", "x_test", "y_train", "y_test"]),
        data_description,
    )
