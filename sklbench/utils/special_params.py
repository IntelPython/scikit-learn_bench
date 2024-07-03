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

from copy import deepcopy
from math import ceil
from typing import Dict, List

import numpy as np
import pandas as pd
from psutil import cpu_count
from sklearn.metrics import euclidean_distances

from ..datasets import dataset_loading_functions
from .bench_case import get_bench_case_value, set_bench_case_value
from .common import convert_to_numpy, flatten_list
from .custom_types import BenchCase, BenchTemplate
from .env import get_numa_cpus_conf
from .logger import logger

SP_VALUE_STR = "[SPECIAL_VALUE]"


def is_special_value(value) -> bool:
    return isinstance(value, str) and value.startswith(SP_VALUE_STR)


def explain_range(range_str: str) -> List:
    def check_range_values_size(range_values: List[int], size: int):
        if len(range_values) != size:
            raise ValueError(
                f"Range contains {len(range_values)} " f"numeric values instead of {size}"
            )

    range_values = range_str.replace("[RANGE]", "").split(":")
    # TODO: add float values
    range_type = range_values[0]
    range_values = list(map(int, range_values[1:]))
    # - add:start{int}:end{int}:step{int} - Arithmetic progression
    #   Sequence: start + step * i <= end
    if range_type == "add":
        check_range_values_size(range_values, 3)
        start, end, step = range_values
        return list(range(start, end + step, step))
    # - mul:current{int}:end{int}:step{int} - Geometric progression
    #   Sequence: current * step <= end
    elif range_type == "mul":
        check_range_values_size(range_values, 3)
        current, end, step = range_values
        result = list()
        while current <= end:
            result.append(current)
            current *= step
        return result
    # - pow:base{int}:start{int}:end{int}[:step{int}] - Powers of base number
    elif range_type == "pow":
        # add default step = 1 if not defined
        if len(range_values) < 4:
            range_values.append(1)
        check_range_values_size(range_values, 4)
        base, start, end, step = range_values
        return [base**i for i in range(start, end + step, step)]
    else:
        raise ValueError(f'Unknown "{range_type}" range type')


def assign_template_special_values(template: BenchTemplate) -> BenchTemplate:
    # data:dataset special values
    datasets = deepcopy(get_bench_case_value(template, "data:dataset"))
    if datasets is not None:
        if not isinstance(datasets, list):
            datasets = [datasets]
        # `all_named` is equal to all datasets known by data loaders
        all_named_datasets = list(dataset_loading_functions.keys())
        for i, dataset in enumerate(datasets):
            if is_special_value(dataset):
                dataset = dataset.replace(SP_VALUE_STR, "")
                if dataset == "all_named":
                    datasets[i] = all_named_datasets
        datasets = flatten_list(datasets, ensure_type_homogeneity=True)
        set_bench_case_value(template, "data:dataset", datasets)

    return template


def assign_case_special_values_on_generation(bench_case: BenchCase) -> BenchCase:
    # sklearn.datasets.make_classification: n_informative as ratio of n_features
    n_informative = get_bench_case_value(
        bench_case, "data:generation_kwargs:n_informative"
    )
    if is_special_value(n_informative):
        n_informative = float(n_informative.replace(SP_VALUE_STR, ""))
        if n_informative <= 0.0 or n_informative > 1.0:
            raise ValueError(f'Wrong special value "{n_informative}" for n_informative')
        n_features = get_bench_case_value(bench_case, "data:generation_kwargs:n_features")
        if n_features is None:
            raise ValueError(
                '"n_features" is not specified for special value of "n_informative"'
            )
        set_bench_case_value(
            bench_case,
            "data:generation_kwargs:n_informative",
            ceil(n_informative * n_features),
        )
    # taskset
    taskset = get_bench_case_value(bench_case, "bench:taskset")
    if is_special_value(taskset):
        taskset = taskset.replace(SP_VALUE_STR, "")
        # special value format for numa nodes: "numa:{numa_node_0}[|{numa_node_1}...]"
        if taskset.startswith("numa"):
            numa_nodes = list(map(int, taskset.split(":")[1].split("|")))
            numa_cpus_conf = get_numa_cpus_conf()
            taskset = ",".join([numa_cpus_conf[numa_node] for numa_node in numa_nodes])
            set_bench_case_value(bench_case, "bench:taskset", taskset)

    # remove requested parameters from the case
    def traverse_with_removal(case: BenchCase):
        for key, value in list(case.items()):
            if isinstance(value, dict):
                traverse_with_removal(value)
            elif isinstance(value, str) and value == "[REMOVE]":
                del case[key]

    traverse_with_removal(bench_case)

    return bench_case


def get_ratio_from_n_jobs(n_jobs: str) -> float:
    args = n_jobs.split(":")
    if len(args) == 1:
        return 1.0
    elif len(args) == 2:
        return float(args[1])
    else:
        raise ValueError(f'Wrong arguments {args} in "n_jobs" special value')


def assign_case_special_values_on_run(
    bench_case: BenchCase, data, data_description: Dict
):
    # Note: data = (x_train, y_train, x_test, y_train)
    library = get_bench_case_value(bench_case, "algorithm:library", None)
    estimator = get_bench_case_value(bench_case, "algorithm:estimator", None)
    # device-related parameters assignment
    device = get_bench_case_value(bench_case, "algorithm:device", "default")
    if device != "default":
        # xgboost tree method assignment branch
        if library == "xgboost" and estimator in ["XGBRegressor", "XGBClassifier"]:
            if device == "cpu" or any(map(device.startswith, ["gpu", "cuda"])):
                logger.debug(
                    f"Forwaring device '{device}' to XGBoost estimator parameters"
                )
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:device", device
                )
            else:
                raise ValueError(f"Unknown device '{device}' for xgboost {estimator}")
        # set target offload for execution context
        elif library.startswith("sklearnex") or library.startswith("daal4py"):
            if device == "cpu":
                logger.debug(
                    "Skipping setting of 'target_offload' for CPU device "
                    "to avoid extra overheads"
                )
            else:
                set_bench_case_value(
                    bench_case, "algorithm:sklearnex_context:target_offload", device
                )
        # faiss GPU algorithm selection
        elif library == "sklbench.emulators.faiss" and estimator == "NearestNeighbors":
            set_bench_case_value(bench_case, "algorithm:estimator_params:device", device)
        else:
            logger.warning(f'Device specification "{device}" is not used for this case')
    # assign "default" or changed device for output
    tree_method = get_bench_case_value(
        bench_case, "algorithm:estimator_params:tree_method", None
    )
    if tree_method == "gpu_hist":
        device = "gpu"
    set_bench_case_value(bench_case, "algorithm:device", device)
    # n_jobs
    n_jobs = get_bench_case_value(bench_case, "algorithm:estimator_params:n_jobs", None)
    if is_special_value(n_jobs):
        n_jobs = n_jobs.replace(SP_VALUE_STR, "")
        if n_jobs.startswith("physical_cpus"):
            n_cpus = cpu_count(logical=False)
        elif n_jobs.startswith("logical_cpus"):
            n_cpus = cpu_count(logical=True)
        else:
            raise ValueError(f'Unknown special value {n_jobs} for "n_jobs"')
        n_jobs = int(n_cpus * get_ratio_from_n_jobs(n_jobs))
        set_bench_case_value(bench_case, "algorithm:estimator_params:n_jobs", n_jobs)
    # classes balance for XGBoost
    scale_pos_weight = get_bench_case_value(
        bench_case, "algorithm:estimator_params:scale_pos_weight", None
    )
    if (
        is_special_value(scale_pos_weight)
        and scale_pos_weight.replace(SP_VALUE_STR, "") == "auto"
        and library == "xgboost"
        and estimator == "XGBClassifier"
    ):
        y_train = convert_to_numpy(data[1])
        value_counts = pd.value_counts(y_train).sort_index()
        if len(value_counts) != 2:
            logger.info(
                f"Number of classes ({len(value_counts)}) != 2 "
                'while "scale_pos_weight" is set to "auto". '
                "This parameter is removed from estimator parameters."
            )
            set_bench_case_value(
                bench_case, "algorithm:estimator_params:scale_pos_weight", None
            )
        else:
            scale_pos_weight = value_counts.iloc[0] / value_counts.iloc[1]
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:scale_pos_weight",
                scale_pos_weight,
            )
    # "n_clusters" auto assignment from data description
    n_clusters = get_bench_case_value(
        bench_case, "algorithm:estimator_params:n_clusters", None
    )
    if is_special_value(n_clusters) and n_clusters.replace(SP_VALUE_STR, "") == "auto":
        n_clusters = data_description.get("n_clusters", None)
        n_classes = data_description.get("n_classes", None)
        n_clusters_per_class = data_description.get("n_clusters_per_class", 1)
        if n_clusters is not None:
            if isinstance(n_clusters, int):
                set_bench_case_value(
                    bench_case, "algorithm:estimator_params:n_clusters", n_clusters
                )
            else:
                raise ValueError(
                    f"n_clusters={n_clusters} of type {type(n_clusters)} "
                    "from data description is not integer."
                )
        elif n_classes is not None:
            set_bench_case_value(
                bench_case,
                "algorithm:estimator_params:n_clusters",
                n_classes * n_clusters_per_class,
            )
        else:
            raise ValueError(
                "Unable to auto-assign n_clusters: "
                "data description doesn't have n_clusters or n_classes"
            )
    # "eps" auto assignment for DBSCAN
    eps = get_bench_case_value(bench_case, "algorithm:estimator_params:eps", None)
    if is_special_value(eps) and eps.replace(SP_VALUE_STR, "").startswith(
        "distances_quantile"
    ):
        x_train = convert_to_numpy(data[0])
        quantile = float(eps.replace(SP_VALUE_STR, "").split(":")[1])
        # subsample of x_train is used to avoid reaching of memory limit for large matrices
        subsample = list(getattr(x_train, "index", np.arange(x_train.shape[0])))
        np.random.seed(42)
        np.random.shuffle(subsample)
        subsample = subsample[: min(x_train.shape[0], 1000)]
        x_sample = (
            x_train.loc[subsample] if hasattr(x_train, "loc") else x_train[subsample]
        )
        # conversion to lower precision is required
        # to produce same distances quantile for different dtypes of x
        x_sample = x_sample.astype("float32")
        dist = np.tril(euclidean_distances(x_sample, x_sample)).reshape(-1)
        dist = dist[dist != 0]
        quantile = float(np.quantile(dist, quantile))
        set_bench_case_value(bench_case, "algorithm:estimator_params:eps", quantile)
