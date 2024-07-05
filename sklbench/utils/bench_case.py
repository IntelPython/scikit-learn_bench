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
from typing import Any, List, Union

from .custom_types import BenchCase, JsonTypesUnion


def set_bench_case_value(
    bench_case: BenchCase, param_name: Union[List[str], str], value: JsonTypesUnion
):
    if isinstance(param_name, str):
        keys_chain = param_name.split(":")
    else:
        keys_chain = param_name
    # deep dive into bench case
    local_value = bench_case
    for prev_key in keys_chain[:-1]:
        if prev_key not in local_value:
            local_value[prev_key] = dict()
        local_value = local_value[prev_key]
    local_value[keys_chain[-1]] = value  # type: ignore


def get_bench_case_value(
    bench_case: BenchCase,
    param_name: Union[List[str], str],
    default_value: JsonTypesUnion = None,
) -> Any:
    if isinstance(param_name, str):
        keys_chain = param_name.split(":")
    else:
        keys_chain = param_name
    # deep dive into bench case
    local_value = bench_case
    for prev_key in keys_chain:
        if prev_key not in local_value:
            return default_value
        local_value = local_value[prev_key]
    return deepcopy(local_value)


def get_bench_case_values(
    bench_case: BenchCase,
    param_names: Union[List[List[str]], List[str]],
    default_value: JsonTypesUnion = None,
) -> List[Any]:
    return list(
        get_bench_case_value(bench_case, param_name, default_value)
        for param_name in param_names
    )


def get_first_of_bench_case_values(
    bench_case: BenchCase, param_names: Union[List[List[str]], List[str]]
) -> JsonTypesUnion:
    values = get_bench_case_values(bench_case, param_names, None)
    values = list(filter(lambda x: x is not None, values))
    if len(values) == 0:
        raise ValueError(f"Unable to find any of values: {param_names}.")
    else:
        return values[0]


def apply_func_to_bench_case_values(
    bench_case: BenchCase, func, copy: bool = False
) -> BenchCase:
    if copy:
        result = deepcopy(bench_case)
    else:
        result = bench_case
    for key, value in result.items():
        if isinstance(value, dict):
            apply_func_to_bench_case_values(value, func)
        else:
            result[key] = func(value)
    return result


def get_data_name(bench_case: BenchCase, shortened: bool = False) -> str:
    # check if unique dataset name is specified directly
    dataset = get_bench_case_value(bench_case, "data:dataset")
    if dataset is not None:
        return dataset
    # check source of data
    source = get_bench_case_value(bench_case, "data:source")
    # generate kwargs postfixes for data filename
    postfixes = dict()
    for kwargs_type in ["generation", "dataset"]:
        postfix = ""
        for key, value in get_bench_case_value(
            bench_case, f"data:{kwargs_type}_kwargs", dict()
        ).items():
            postfix += f"_{key}_{value}"
        postfixes[kwargs_type] = postfix
    # fetch_openml
    if source == "fetch_openml":
        openml_id = get_bench_case_value(bench_case, "data:id")
        return f"openml_{openml_id}"
    # make_*
    if source in ["make_classification", "make_regression", "make_blobs"]:
        name = source
        if shortened:
            return name.replace("classification", "clsf").replace("regression", "regr")
        else:
            return f'{name}{postfixes["generation"]}{postfixes["dataset"]}'
    raise ValueError("Unable to get data name")


def get_bench_case_name(
    bench_case: BenchCase, shortened: bool = False, separator: str = " "
) -> str:
    library_name = get_bench_case_value(bench_case, "algorithm:library")
    alg_name = get_first_of_bench_case_values(
        bench_case, ["algorithm:estimator", "algorithm:function"]
    )
    data_name = get_data_name(bench_case, shortened)
    name_args = [library_name, alg_name, data_name]
    device = get_bench_case_value(bench_case, "algorithm:device", None)
    if device is not None:
        name_args.append(device)
    return separator.join(name_args)
