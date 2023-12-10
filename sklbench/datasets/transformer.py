# ===============================================================================
# Copyright 2023 Intel Corporation
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

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from ..utils.bench_case import get_bench_case_value


def convert_data(data, dformat: str, order: str, dtype: str):
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

        converted_data = convert_data(subset_content, data_format, data_order, data_dtype)
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
