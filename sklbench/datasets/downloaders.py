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
import time
from typing import Callable, List, Union

import numpy as np
import openml
import pandas as pd
import requests
from scipy.sparse import csr_matrix


def retrieve(url: str, filename: str, max_retries: int = 3) -> None:
    """Download a file from a URL with basic retry logic."""
    if os.path.isfile(filename):
        return

    if not url.startswith("http"):
        raise ValueError(f"URL must start with http:// or https://, got: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=120)
            if response.status_code != 200:
                raise AssertionError(
                    f"Failed to download from {url}. "
                    f"Response returned status code {response.status_code}"
                )

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with open(filename, "wb") as datafile:
                bytes_written = 0
                for data in response.iter_content(block_size):
                    if data:
                        datafile.write(data)
                        bytes_written += len(data)

            # Verify download completeness if size is known
            if total_size > 0 and bytes_written != total_size:
                os.remove(filename)
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise AssertionError(
                    f"Incomplete download from {url}. "
                    f"Expected {total_size} bytes, got {bytes_written}"
                )
            return

        except (
            requests.exceptions.RequestException,
            IOError,
        ) as e:
            if os.path.isfile(filename):
                os.remove(filename)
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise AssertionError(
                f"Failed to download {url} after {max_retries} attempts: {e}"
            ) from e


def fetch_and_correct_openml(
    data_id: int, raw_data_cache_dir: str, as_frame: str = "auto"
):
    """Fetch OpenML dataset using the openml package."""
    # Configure openml cache directory
    openml_cache = os.path.join(raw_data_cache_dir, "openml")
    os.makedirs(openml_cache, exist_ok=True)
    openml.config.set_root_cache_directory(openml_cache)

    # Fetch the dataset
    dataset = openml.datasets.get_dataset(
        data_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )

    # Get the data with target column specified
    x, y, _, _ = dataset.get_data(
        dataset_format="dataframe" if as_frame is True else "array",
        target=dataset.default_target_attribute,
    )

    # Validate x type
    if not isinstance(x, (csr_matrix, pd.DataFrame, np.ndarray)):
        raise ValueError(f'Unknown x type "{type(x)}" returned from openml')

    # Convert sparse DataFrame to dense format
    if isinstance(x, pd.DataFrame):
        if any(pd.api.types.is_sparse(x[col]) for col in x.columns):
            x = x.sparse.to_dense()

    # Convert y to numpy array if needed
    if isinstance(y, pd.Series):
        if isinstance(y.dtype, pd.CategoricalDtype):
            y = y.cat.codes
        # Use to_numpy() for sparse arrays to densify them, otherwise use values
        if pd.api.types.is_sparse(y):
            y = y.to_numpy()
        else:
            y = y.values
    elif not isinstance(y, np.ndarray):
        raise ValueError(f'Unknown y type "{type(y)}" returned from openml')

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
