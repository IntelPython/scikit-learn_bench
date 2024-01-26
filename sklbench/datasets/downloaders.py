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
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_openml

try:
    import kaggle

    kaggle_is_imported = True
except (ImportError, OSError, ValueError):
    kaggle_is_imported = False


def retrieve(url: str, filename: str) -> None:
    if os.path.isfile(filename):
        return
    elif url.startswith("http"):
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise AssertionError(
                f"Failed to download from {url}.\n"
                f"Response returned status code {response.status_code}"
            )
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        n = 0
        with open(filename, "wb+") as datafile:
            for data in response.iter_content(block_size):
                n += len(data) / 1024
                datafile.write(data)
        if total_size != 0 and n != total_size / 1024:
            raise AssertionError("Some content was present but not downloaded/written")


def fetch_and_correct_openml(
    data_id: int, raw_data_cache_dir: str, as_frame: str = "auto"
):
    x, y = fetch_openml(
        data_id=data_id, return_X_y=True, as_frame=as_frame, data_home=raw_data_cache_dir
    )
    if (
        isinstance(x, csr_matrix)
        or isinstance(x, pd.DataFrame)
        or isinstance(x, np.ndarray)
    ):
        pass
    else:
        raise ValueError(f'Unknown "{type(x)}" x type was returned from fetch_openml')
    if isinstance(y, pd.Series):
        # label transforms to cat.codes if it is passed as categorical series
        if isinstance(y.dtype, pd.CategoricalDtype):
            y = y.cat.codes
        y = y.values
    elif isinstance(y, np.ndarray):
        pass
    else:
        raise ValueError(f'Unknown "{type(y)}" y type was returned from fetch_openml')
    return x, y


def load_openml(
    data_id: int,
    raw_data_cache_dir: str,
    transform_x_y_func: Union[Callable, None] = None,
    as_frame: str = "auto",
):
    x, y = fetch_and_correct_openml(data_id, raw_data_cache_dir, as_frame)
    if transform_x_y_func is not None:
        x, y = transform_x_y_func(x, y)
    return x, y


def download_and_read_csv(url: str, raw_data_cache_dir: str, **reading_kwargs):
    local_path = os.path.join(raw_data_cache_dir, os.path.basename(url))
    retrieve(url, local_path)
    data = pd.read_csv(local_path, **reading_kwargs)
    return data


def download_kaggle_files(
    kaggle_type: str, kaggle_name: str, filenames: List[str], raw_data_cache_dir: str
):
    if not kaggle_is_imported:
        raise ValueError("Kaggle API is not available.")
    api = kaggle.KaggleApi()
    api.authenticate()

    if kaggle_type == "competition":
        download_method = api.competition_download_file
    elif kaggle_type == "dataset":
        download_method = api.dataset_download_file
    else:
        raise ValueError(
            f"Unknown {kaggle_type} type for " '"download_kaggle_files" function.'
        )

    output_file_paths = {}
    for filename in filenames:
        download_method(kaggle_name, filename, raw_data_cache_dir)
        output_file_paths[filename] = os.path.join(raw_data_cache_dir, filename)
    return output_file_paths
