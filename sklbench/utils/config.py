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

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Union

from .bench_case import get_bench_case_value, set_bench_case_value
from .common import (
    convert_to_numeric_if_possible,
    custom_format,
    flatten_list,
    hash_from_json_repr,
)
from .custom_types import BenchCase, BenchTemplate
from .logger import logger
from .special_params import (
    assign_case_special_values_on_generation,
    assign_template_special_values,
    explain_range,
)


def find_configs(paths: Union[List[str], str, None]) -> List[str]:
    result = list()
    if paths is None:
        return result
    # iterate over list of paths
    elif isinstance(paths, list):
        for path in paths:
            result += find_configs(path)
    # check if path is *.json file
    elif os.path.isfile(paths) and paths.endswith(".json"):
        result.append(paths)
    # iterate over directory content with recursion
    elif os.path.isdir(paths):
        for path in os.listdir(paths):
            result += find_configs(os.path.join(paths, path))
    else:
        logger.debug(f'Config path "{paths}" wasn\'t added')
    return result


def merge_dicts(first: Dict, second: Dict) -> Dict:
    # Function overwrites deep copy of first with second
    # `deepcopy` is used to avoid accidental changes
    # through reference to list or dict
    result = deepcopy(first)
    # iteration over items of second dict with inner recursion
    for key, value in second.items():
        if key not in result:
            result[key] = deepcopy(value)
        else:
            # `dict | dict` case - simple merge
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, dict):
                result[key] = [merge_dicts(el, deepcopy(value)) for el in result[key]]
            elif isinstance(result[key], dict) and isinstance(value, list):
                result[key] = [merge_dicts(result[key], deepcopy(el)) for el in value]
            elif isinstance(result[key], list) and isinstance(value, list):
                local_result = []
                for element_in_first in result[key]:
                    for element_in_second in value:
                        local_result.append(
                            merge_dicts(element_in_first, element_in_second)
                        )
                result[key] = local_result
            else:
                result[key] = deepcopy(value)
    return result


def parse_config_file(config_path: str) -> List[Dict]:
    with open(config_path, "r") as config_file:
        config_content = json.load(config_file)
    templates = list()
    if "TEMPLATES" not in config_content:
        raise ValueError(f"{config_path} doesn't contain templates")
    if "INCLUDE" in config_content:
        config_dir = os.path.dirname(config_path)
        include_content = dict()
        for include_config in config_content["INCLUDE"]:
            include_path = os.path.join(config_dir, include_config)
            if os.path.isfile(include_path):
                with open(include_path, "r") as include_file:
                    include_content.update(json.load(include_file)["PARAMETERS_SETS"])
            else:
                logger.warning(f"Include file '{include_path}' not found.")
        include_content.update(config_content["PARAMETERS_SETS"])
        config_content["PARAMETERS_SETS"] = include_content
    for template_name, template_content in config_content["TEMPLATES"].items():
        new_templates = [{}]
        # 1st step: pop list of included param sets and add them to template
        if "SETS" in template_content:
            for param_set_name in template_content.pop("SETS"):
                param_set = config_content["PARAMETERS_SETS"][param_set_name]
                if isinstance(param_set, dict):
                    new_templates = [
                        merge_dicts(tmpl, param_set) for tmpl in new_templates
                    ]
                elif isinstance(param_set, list):
                    new_templates = flatten_list(
                        [
                            [merge_dicts(tmpl, set_element) for set_element in param_set]
                            for tmpl in new_templates
                        ]
                    )
        # 2nd step: add other params for specific template
        new_templates = [merge_dicts(tmpl, template_content) for tmpl in new_templates]
        templates += new_templates
    return templates


def parse_cli_parameters(params: List) -> BenchTemplate:
    result = dict()
    for param in params:
        # parameter format: "key1:key2:key3=value1[,value2]"
        param = param.split("=")
        param_path, param_values = param[0].split(":"), param[1]
        param_values = param_values.split(",")
        # int/float/bool/None values are initially read as str
        for i, value in enumerate(param_values):
            if param_values[i] == "null":
                param_values[i] = None
            elif param_values[i] == "true":
                param_values[i] = True
            elif param_values[i] == "false":
                param_values[i] = False
            else:
                param_values[i] = convert_to_numeric_if_possible(value)
        if len(param_values) == 1:
            param_values = param_values[0]
        # deduce chain of param keys
        param_dict = dict()
        local_dict = param_dict
        for key in param_path[:-1]:
            local_dict[key] = dict()
            local_dict = local_dict[key]
        local_dict[param_path[-1]] = param_values
        result = merge_dicts(result, param_dict)

    return result


def expand_ranges_in_template(template: BenchTemplate):
    for key, value in template.items():
        # recursion for inner dict
        if isinstance(value, dict):
            expand_ranges_in_template(value)
        # iteration over list values
        elif isinstance(value, list):
            for i, el in enumerate(value):
                # list of dicts
                if isinstance(el, dict):
                    expand_ranges_in_template(el)
                # list of strs
                elif isinstance(el, str) and el.startswith("[RANGE]"):
                    value[i] = explain_range(el)
            # avoidance of nested lists
            # (in bench_case where ranges and strs are mixed)
            template[key] = flatten_list(value)
        elif isinstance(value, str) and value.startswith("[RANGE]"):
            template[key] = explain_range(value)
            if len(template[key]) == 0:
                raise ValueError("Range specification resulted in zero-length list")


def expand_template(
    template: BenchTemplate, bench_cases: List[Dict], keys_chain: List[str]
) -> List[Dict]:
    # deep copy to prevent modifying by reference
    bench_cases = deepcopy(bench_cases)
    # iterate over dict
    if isinstance(template, dict):
        for key, value in template.items():
            bench_cases = expand_template(value, bench_cases, keys_chain + [key])
    # iterate over list
    elif isinstance(template, list):
        new_bench_cases = list()
        for i, value in enumerate(template):
            new_bench_cases += expand_template(value, bench_cases, keys_chain)
        bench_cases = new_bench_cases
    # assign scalar value
    else:
        for bench_case in bench_cases:
            set_bench_case_value(bench_case, keys_chain, template)
    return bench_cases


def remove_duplicated_bench_cases(bench_cases: List[BenchCase]) -> List[BenchCase]:
    hash_map = dict()
    for bench_case in bench_cases:
        hash_map[hash_from_json_repr(bench_case)] = bench_case
    return list(hash_map.values())


def bench_case_filter(bench_case: BenchCase, filters: List[BenchCase]):
    # filtering is implemented by comparison of
    # benchmark case and case merged with filters:
    # filtering is passed if one of merged cases has same hash as original
    original_hash = hash_from_json_repr(bench_case)
    filtered_hashes = [
        hash_from_json_repr(merge_dicts(bench_case, bench_filter))
        for bench_filter in filters
    ]
    return original_hash in filtered_hashes or len(filtered_hashes) == 0


def early_filtering(
    bench_cases: List[BenchCase], filters: List[BenchCase]
) -> List[BenchCase]:
    def get_early_filter(original_filter):
        static_params = [
            "data",
            "algorithm:library",
            "algorithm:estimator",
            "algorithm:function",
            "algorithm:device",
        ]
        early_filter = dict()
        for static_param in static_params:
            early_value = get_bench_case_value(original_filter, static_param)
            if early_value is not None:
                set_bench_case_value(early_filter, static_param, early_value)
        return early_filter

    static_param_filters = list(map(get_early_filter, filters))
    filtered_bench_cases = list(
        filter(lambda x: bench_case_filter(x, static_param_filters), bench_cases)
    )
    if len(bench_cases) != len(filtered_bench_cases):
        logger.info(
            "Early filtering reduced number of cases from "
            f"{len(bench_cases)} to {len(filtered_bench_cases)}."
        )
    return filtered_bench_cases


def generate_bench_filters(raw_filters: List) -> List[BenchCase]:
    # filters are implemented as benchmark cases
    # containing only filter values
    filters_template = parse_cli_parameters(raw_filters)
    filters_template = assign_template_special_values(filters_template)
    expand_ranges_in_template(filters_template)
    filters = expand_template(filters_template, [{}], [])
    filters = remove_duplicated_bench_cases(filters)
    filters = list(map(assign_case_special_values_on_generation, filters))
    logger.debug(f"Loaded filters:\n{custom_format(filters)}")
    return filters


def generate_bench_cases(args: argparse.Namespace) -> List[BenchCase]:
    # find config files from paths specified in args
    config_files = find_configs(args.config)

    # config files or global parameters should be defined for run
    if len(config_files) == 0:
        if args.parameters == "":
            raise ValueError("Unable to find any configs")
        else:
            logger.info("Using CLI parameters as template")
    else:
        logger.info(f"Number of found config files: {len(config_files)}")
        logger.debug(f"Found config files:\n{custom_format(config_files)}")

    # parse config files to get bench_case templates
    # (without expanded paramaters from lists and ranges)
    bench_case_templates = list()
    for config_file in config_files:
        bench_case_templates += parse_config_file(config_file)

    # overwrite templates by globally defined parameters or use them as template
    global_parameters = parse_cli_parameters(args.parameters)
    if len(global_parameters) > 0:
        logger.info(f"Global parameters:\n{custom_format(global_parameters)}")
    else:
        logger.debug("Global parameters are empty")
    if len(bench_case_templates) == 0:
        bench_case_templates = [global_parameters]
    else:
        bench_case_templates = [
            merge_dicts(tmpl, global_parameters) for tmpl in bench_case_templates
        ]
        logger.info(f"Number of loaded templates: {len(bench_case_templates)}")
        logger.debug(f"Loaded templates:\n{custom_format(bench_case_templates)}")

    # assign special values in templates
    bench_case_templates = list(map(assign_template_special_values, bench_case_templates))

    # extract values from lists and ranges defined in templates
    for tmpl in bench_case_templates:
        expand_ranges_in_template(tmpl)

    all_bench_cases = list()
    # find non-duplicated bench_cases from templates
    for tmpl in bench_case_templates:
        all_bench_cases += expand_template(tmpl, [{}], [])
    logger.debug(
        f"Number of loaded cases before removal of duplicates: {len(all_bench_cases)}"
    )
    all_bench_cases = remove_duplicated_bench_cases(all_bench_cases)

    # assign special values in bench_cases
    all_bench_cases = list(map(assign_case_special_values_on_generation, all_bench_cases))

    logger.info(f"Number of loaded cases: {len(all_bench_cases)}")
    logger.debug(f"Loaded cases:\n{custom_format(all_bench_cases)}")

    return all_bench_cases
