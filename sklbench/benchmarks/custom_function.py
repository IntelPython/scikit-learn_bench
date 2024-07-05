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

from typing import Dict, List, Tuple

from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.bench_case import get_bench_case_value
from ..utils.common import get_module_members
from ..utils.config import bench_case_filter
from ..utils.custom_types import BenchCase
from ..utils.logger import logger
from ..utils.measurement import measure_case
from ..utils.special_params import assign_case_special_values_on_run
from .common import enrich_result, main_template


def get_function_instance(library_name: str, function_name: str):
    _, functions_map = get_module_members(library_name.split("."))
    if function_name not in functions_map:
        raise ValueError(
            f"Unable to find {function_name} function in {library_name} module."
        )
    if len(functions_map[function_name]) != 1:
        logger.debug(
            f'List of estimator with name "{function_name}": '
            f"{functions_map[function_name]}"
        )
        logger.warning(
            f"Found {len(functions_map[function_name])} classes for "
            f'"{function_name}" estimator name. '
            f"Using first {functions_map[function_name][0]}."
        )
    return functions_map[function_name][0]


def get_function_args(bench_case: BenchCase, x_train, y_train, x_test, y_test) -> Tuple:
    args_map = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
    # order format: "arg1|arg2|...|argN"
    args_order = get_bench_case_value(
        bench_case, "algorithm:args_order", "x_train|y_train"
    )
    args = (args_map[arg] for arg in args_order.split("|"))
    return args


def measure_function_instance(bench_case, function_instance, args: Tuple, kwargs: Dict):
    metrics = dict()
    metrics["time[ms]"], metrics["time std[ms]"], _ = measure_case(
        bench_case, function_instance, *args, **kwargs
    )
    return metrics


def main(bench_case: BenchCase, filters: List[BenchCase]):
    library_name = get_bench_case_value(bench_case, "algorithm:library")
    function_name = get_bench_case_value(bench_case, "algorithm:function")

    function_instance = get_function_instance(library_name, function_name)

    # load and transform data
    data, data_description = load_data(bench_case)
    (x_train, x_test, y_train, y_test), data_description = split_and_transform_data(
        bench_case, data, data_description
    )

    # assign special values
    assign_case_special_values_on_run(
        bench_case, (x_train, y_train, x_test, y_test), data_description
    )

    function_args = get_function_args(bench_case, x_train, y_train, x_test, y_test)

    if not bench_case_filter(bench_case, filters):
        logger.warning("Benchmarking case was filtered.")
        return list()

    metrics = measure_function_instance(
        bench_case,
        function_instance,
        function_args,
        get_bench_case_value(bench_case, "algorithm:kwargs", dict()),
    )
    result = {
        "task": "utility",
        "function": function_name,
    }
    result = enrich_result(result, bench_case)
    # TODO: replace `x_train` data_desc with more informative values
    result.update(data_description["x_train"])
    result.update(metrics)
    return [result]


if __name__ == "__main__":
    main_template(main)
