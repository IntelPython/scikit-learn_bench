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
import warnings
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_openml


def retrieve(url: str, filename: str, max_retries: int = 5) -> None:
    """
    Download a file from a URL with retry logic and resume capability.

    Args:
        url: URL to download from
        filename: Local file path to save to
        max_retries: Maximum number of retry attempts for failed downloads
    """
    if os.path.isfile(filename):
        # Check if file is complete by comparing size
        try:
            head_response = requests.head(url, allow_redirects=True, timeout=30)
            expected_size = int(head_response.headers.get("content-length", 0))
            actual_size = os.path.getsize(filename)

            if expected_size > 0 and actual_size == expected_size:
                # File exists and is complete
                return
            else:
                warnings.warn(
                    f"Existing file {filename} is incomplete ({actual_size}/{expected_size} bytes). "
                    f"Will attempt to resume download.",
                    RuntimeWarning
                )
        except Exception as e:
            # If we can't verify, assume file is complete
            warnings.warn(
                f"Could not verify file completeness for {filename}: {e}. Assuming complete.",
                RuntimeWarning
            )
            return

    if not url.startswith("http"):
        raise ValueError(f"URL must start with http:// or https://, got: {url}")

    temp_filename = filename + ".partial"
    block_size = 8192

    for attempt in range(max_retries):
        try:
            # Check if we can resume a partial download
            resume_pos = 0
            if os.path.isfile(temp_filename):
                resume_pos = os.path.getsize(temp_filename)
                headers = {"Range": f"bytes={resume_pos}-"}
                mode = "ab"  # Append mode
                warnings.warn(
                    f"Resuming download of {url} from byte {resume_pos}",
                    RuntimeWarning
                )
            else:
                headers = {}
                mode = "wb"

            response = requests.get(url, stream=True, headers=headers, timeout=60)

            # Handle different response codes
            if response.status_code == 200:
                # Full download
                mode = "wb"
                resume_pos = 0
            elif response.status_code == 206:
                # Partial content (resume successful)
                pass
            elif response.status_code == 416:
                # Range not satisfiable - file might be complete
                if os.path.isfile(temp_filename):
                    os.rename(temp_filename, filename)
                return
            else:
                raise AssertionError(
                    f"Failed to download from {url}. "
                    f"Response returned status code {response.status_code}"
                )

            # Get expected total size
            if response.status_code == 206:
                content_range = response.headers.get("content-range", "")
                if content_range:
                    total_size = int(content_range.split("/")[1])
                else:
                    total_size = 0
            else:
                total_size = int(response.headers.get("content-length", 0))

            # Download the file
            bytes_downloaded = resume_pos
            with open(temp_filename, mode) as datafile:
                for data in response.iter_content(block_size):
                    if data:  # filter out keep-alive chunks
                        datafile.write(data)
                        bytes_downloaded += len(data)

            # Verify download completeness
            if total_size > 0:
                actual_size = os.path.getsize(temp_filename)
                if actual_size != total_size:
                    warnings.warn(
                        f"Download incomplete: {actual_size}/{total_size} bytes. "
                        f"Attempt {attempt + 1}/{max_retries}",
                        RuntimeWarning
                    )
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        raise AssertionError(
                            f"Failed to completely download {url} after {max_retries} attempts. "
                            f"Got {actual_size}/{total_size} bytes"
                        )

            # Download successful, rename temp file to final filename
            os.rename(temp_filename, filename)
            return

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            warnings.warn(
                f"Download interrupted for {url}: {type(e).__name__}: {e}. "
                f"Attempt {attempt + 1}/{max_retries}",
                RuntimeWarning
            )
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                warnings.warn(f"Waiting {wait_time}s before retry...", RuntimeWarning)
                time.sleep(wait_time)
                continue
            else:
                # Clean up partial file if all retries failed
                if os.path.isfile(temp_filename):
                    os.remove(temp_filename)
                raise AssertionError(
                    f"Failed to download {url} after {max_retries} attempts. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e


def fetch_and_correct_openml(
    data_id: int, raw_data_cache_dir: str, as_frame: str = "auto"
):
    """
    Fetch OpenML dataset with fallback for MD5 checksum errors.

    First tries sklearn's fetch_openml. If that fails due to MD5 checksum mismatch,
    falls back to using the openml package directly, which has updated checksums.
    """
    try:
        # Try sklearn's fetch_openml first
        x, y = fetch_openml(
            data_id=data_id, return_X_y=True, as_frame=as_frame, data_home=raw_data_cache_dir
        )
    except ValueError as e:
        # Check if it's an MD5 checksum error
        if "md5 checksum" in str(e).lower():
            warnings.warn(
                f"MD5 checksum validation failed for OpenML dataset {data_id}. "
                f"Falling back to using openml package directly. "
                f"Original error: {e}",
                RuntimeWarning
            )

            # Fall back to openml package which might have updated checksums
            try:
                import openml
                # Configure openml to use the provided cache directory
                openml_cache = os.path.join(raw_data_cache_dir, "openml_direct")
                os.makedirs(openml_cache, exist_ok=True)
                openml.config.set_root_cache_directory(openml_cache)

                dataset = openml.datasets.get_dataset(
                    data_id,
                    download_data=True,
                    download_qualities=False,
                    download_features_meta_data=False
                )
                #Get the data with target column specified
                x, y, _, _ = dataset.get_data(
                    dataset_format="dataframe" if as_frame == "auto" or as_frame else "array",
                    target=dataset.default_target_attribute
                )
            except Exception as openml_error:
                raise ValueError(
                    f"Failed to load OpenML dataset {data_id} using both sklearn and openml package. "
                    f"sklearn error: {e}. openml error: {openml_error}"
                ) from openml_error
        else:
            # Not a checksum error, re-raise
            raise

    # Validate and convert return types
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
