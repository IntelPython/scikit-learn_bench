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
from typing import Dict, Tuple

from ..utils.bench_case import get_bench_case_value, get_data_name
from ..utils.common import custom_format
from ..utils.custom_types import BenchCase
from .loaders import (
    dataset_loading_functions,
    load_custom_data,
    load_openml_data,
    load_sklearn_synthetic_data,
)


def load_data(bench_case: BenchCase) -> Tuple[Dict, Dict]:
    # get data name and cache dirs
    data_name = get_data_name(bench_case, shortened=False)
    data_cache = get_bench_case_value(bench_case, "data:cache_directory", "data_cache")
    raw_data_cache = get_bench_case_value(
        bench_case, "data:raw_cache_directory", os.path.join(data_cache, "raw")
    )
    common_kwargs = {
        "data_name": data_name,
        "data_cache": data_cache,
        "raw_data_cache": raw_data_cache,
    }
    preproc_kwargs = get_bench_case_value(bench_case, "data:preprocessing_kwargs", dict())
    # make cache directories
    os.makedirs(data_cache, exist_ok=True)
    os.makedirs(raw_data_cache, exist_ok=True)
    # load by dataset name
    dataset = get_bench_case_value(bench_case, "data:dataset")
    if dataset is not None:
        dataset_params = get_bench_case_value(bench_case, "data:dataset_kwargs", dict())
        if dataset in dataset_loading_functions:
            # registered dataset loading branch
            return dataset_loading_functions[dataset](
                **common_kwargs,
                preproc_kwargs=preproc_kwargs,
                dataset_params=dataset_params,
            )
        else:
            # user-provided dataset loading branch
            return load_custom_data(**common_kwargs, preproc_kwargs=preproc_kwargs)

    # load by source
    source = get_bench_case_value(bench_case, "data:source")
    if source is not None:
        # sklearn.datasets functions
        if source.startswith("make_"):
            generation_kwargs = get_bench_case_value(
                bench_case, "data:generation_kwargs", dict()
            )
            return load_sklearn_synthetic_data(
                function_name=source,
                input_kwargs=generation_kwargs,
                preproc_kwargs=preproc_kwargs,
                **common_kwargs,
            )
        # openml dataset
        elif source == "fetch_openml":
            openml_id = get_bench_case_value(bench_case, "data:id")
            return load_openml_data(
                openml_id=openml_id, preproc_kwargs=preproc_kwargs, **common_kwargs
            )

    raise ValueError(
        "Unable to get data from bench_case:\n"
        f'{custom_format(get_bench_case_value(bench_case, "data"))}'
    )
