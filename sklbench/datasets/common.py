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

import json
import os
import re
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ..utils.custom_types import Array
from ..utils.logger import logger

# NB: non-registered data components and extensions will not be found by loader
KNOWN_DATA_COMPONENTS = ["x", "y"]
KNOWN_DATA_EXTENSIONS = ["parq", "npz", "csr.npz"]


def get_expr_by_prefix(prefix: str) -> str:
    def get_or_expr_from_list(a: List[str]) -> str:
        # transforms list to OR expression: "['x', 'y']" -> "x|y"
        return str(a)[1:-1].replace("'", "").replace(", ", "|")

    data_comp_expr = get_or_expr_from_list(KNOWN_DATA_COMPONENTS)
    data_ext_expr = get_or_expr_from_list(KNOWN_DATA_EXTENSIONS)

    return f"{prefix}_({data_comp_expr}).({data_ext_expr})"


def get_filenames_by_prefix(directory: str, prefix: str) -> List[str]:
    assert os.path.isdir(directory)
    prefix_expr = get_expr_by_prefix(prefix)
    return list(
        filter(lambda x: re.search(prefix_expr, x) is not None, os.listdir(directory))
    )


def load_data_file(filepath, extension):
    if extension == "parq":
        data = pd.read_parquet(filepath)
    elif extension.endswith("npz"):
        npz_content = np.load(filepath)
        if extension == "npz":
            data = npz_content["arr_0"]
        elif extension == "csr.npz":
            data = csr_matrix(
                tuple(npz_content[attr] for attr in ["data", "indices", "indptr"])
            )
        else:
            raise ValueError(f'Unknown npz subextension "{extension}"')
        npz_content.close()
    else:
        raise ValueError(f'Unknown extension "{extension}"')
    return data


def load_data_from_cache(data_cache: str, data_name: str) -> Dict:
    # data filename format:
    # {data_name}_{data_component}.{file_ext}
    data_filenames = get_filenames_by_prefix(data_cache, data_name)
    data = dict()
    for data_filename in data_filenames:
        if data_filename.endswith(".json"):
            continue
        postfix = data_filename.replace(data_name, "")[1:]
        component, file_ext = postfix.split(".", 1)
        data[component] = load_data_file(
            os.path.join(data_cache, data_filename), file_ext
        )
    return data


def save_data_to_cache(data: Dict, data_cache: str, data_name: str):
    for component_name, data_compoment in data.items():
        component_filepath = os.path.join(data_cache, f"{data_name}_{component_name}")
        # convert 2d numpy array to pandas DataFrame for better caching
        if isinstance(data_compoment, np.ndarray) and data_compoment.ndim == 2:
            data_compoment = pd.DataFrame(data_compoment)
        # branching by data type for saving to cache
        if isinstance(data_compoment, pd.DataFrame):
            component_filepath += ".parq"
            data_compoment.columns = [
                column if isinstance(column, str) else str(column)
                for column in list(data_compoment.columns)
            ]
            data_compoment.to_parquet(
                component_filepath, engine="fastparquet", compression="snappy"
            )
        elif isinstance(data_compoment, csr_matrix):
            component_filepath += ".csr.npz"
            np.savez(
                component_filepath,
                **{
                    attr: getattr(data_compoment, attr)
                    for attr in ["data", "indices", "indptr"]
                },
            )
        elif isinstance(data_compoment, pd.Series):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment.to_numpy())
        elif isinstance(data_compoment, np.ndarray):
            component_filepath += ".npz"
            np.savez(component_filepath, data_compoment)


def load_data_description(data_cache: str, data_name: str) -> Dict:
    with open(os.path.join(data_cache, f"{data_name}.json"), "r") as desc_file:
        data_desc = json.load(desc_file)
    return data_desc


def save_data_description(data_desc: Dict, data_cache: str, data_name: str):
    with open(os.path.join(data_cache, f"{data_name}.json"), "w") as desc_file:
        json.dump(data_desc, desc_file)


def cache(function):
    def cache_wrapper(**kwargs):
        data_name = kwargs["data_name"]
        data_cache = kwargs["data_cache"]
        if len(get_filenames_by_prefix(data_cache, data_name)) > 0:
            logger.info(f'Loading "{data_name}" dataset from cache files')
            data = load_data_from_cache(data_cache, data_name)
            data_desc = load_data_description(data_cache, data_name)
        else:
            logger.info(f'Loading "{data_name}" dataset from scratch')
            data, data_desc = function(**kwargs)
            save_data_to_cache(data, data_cache, data_name)
            save_data_description(data_desc, data_cache, data_name)
        return data, data_desc

    return cache_wrapper


def preprocess_data(
    data_dict: List[Dict[str, Array]],
    subsample: Union[float, int, None] = None,
    **kwargs,
) -> List[Dict[str, Array]]:
    """Preprocessing function applied for all data arguments."""
    if subsample is not None:
        for data_name, data in data_dict.items():
            data_dict[data_name] = train_test_split(
                data, train_size=subsample, random_state=42, shuffle=True
            )[0]
    return data_dict


def preprocess_x(
    x: Array,
    replace_nan="auto",
    category_encoding="ordinal",
    normalize=False,
    force_for_sparse=True,
    **kwargs,
) -> Array:
    """Preprocessing function applied only for `x` data argument."""
    return_type = type(x)
    if force_for_sparse and isinstance(x, csr_matrix):
        x = x.toarray()
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if not isinstance(x, pd.DataFrame):
        logger.warning(
            "Preprocessing is supported only for pandas DataFrames "
            f"and numpy ndarray. Got {type(x)} instead."
        )
        return x
    # NaN values replacement
    if x.isna().any().any():
        nan_columns = x.columns[x.isna().any(axis=0)]
        nan_df = x[nan_columns]
        if replace_nan == "auto":
            replace_nan = "median"
            logger.debug(f'Changing "replace_nan" from "auto" to "{replace_nan}".')
        if replace_nan == "median":
            nan_df = nan_df.fillna(nan_df.median())
        elif replace_nan == "mean":
            nan_df = nan_df.fillna(nan_df.mean())
        elif replace_nan == "ignore":
            pass
        else:
            logger.warning(f'Unknown "{replace_nan}" replace nan type.')
        x[nan_columns] = nan_df
    # Categorical features transformation
    categ_columns = x.columns[(x.dtypes == "category") + (x.dtypes == "object")]
    if len(categ_columns) > 0:
        if category_encoding == "onehot":
            prev_n_columns = x.shape[1]
            x = pd.get_dummies(x, columns=list(categ_columns))
            logger.debug(
                f"OneHotEncoder extended {prev_n_columns} columns to {x.shape[1]}."
            )
        elif category_encoding == "ordinal":
            encoder = OrdinalEncoder()
            encoder.set_output(transform="pandas")
            ordinal_df = encoder.fit_transform(x[categ_columns])
            x = x.drop(columns=categ_columns).join(ordinal_df)
        elif category_encoding == "drop":
            x = x.drop(columns=categ_columns)
        elif category_encoding == "ignore":
            pass
        else:
            logger.warning(f'Unknown "{category_encoding}" category encoding type.')
    # Mean-Standard normalization
    if normalize:
        x = (x - x.mean()) / x.std()
    if return_type == np.ndarray:
        return x.values
    else:
        return x


def preprocess(function):
    def preprocess_wrapper(**kwargs):
        preproc_kwargs = kwargs.pop("preproc_kwargs", dict())
        data, data_desc = function(**kwargs)
        data = preprocess_data(data, **preproc_kwargs)
        data["x"] = preprocess_x(data["x"], **preproc_kwargs)
        return data, data_desc

    return preprocess_wrapper
