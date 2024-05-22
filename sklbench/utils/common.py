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

import hashlib
import importlib
import inspect
import json
import re
import subprocess as sp
from pprint import pformat
from shutil import get_terminal_size
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .custom_types import JsonTypesUnion, ModuleContentMap, Numeric

# ANSI escape codes for in-terminal formatting
BCOLORS = {
    "FAIL": "\033[91m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "OKBLUE": "\033[94m",
    "HEADER": "\033[95m",
    "OKCYAN": "\033[96m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


def custom_format(
    input_obj: Any,
    bcolor: Union[str, None] = None,
    prettify: bool = True,
    width: int = get_terminal_size().columns,
    indent: int = 4,
) -> str:
    """Pretty format with terminal highlighting"""
    output = input_obj.copy() if hasattr(input_obj, "copy") else input_obj
    if prettify:
        output = pformat(input_obj, width=width, indent=indent)
    if bcolor is not None:
        output = BCOLORS[bcolor] + str(input_obj) + BCOLORS["ENDC"]
    return output


def read_output_from_command(command: str) -> Tuple[int, str, str]:
    """Executes command and returns code, stdout and stderr"""
    res = sp.run(
        command.split(" "),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        encoding="utf-8",
    )
    return res.returncode, res.stdout[:-1], res.stderr[:-1]


def hash_from_json_repr(x: JsonTypesUnion, hash_limit: int = 5) -> str:
    h = hashlib.sha256()
    h.update(bytes(json.dumps(x), encoding="utf-8"))
    return h.hexdigest()[:hash_limit]


def ensure_list_types_homogeneity(input_list: List):
    list_types = set([type(el) for el in input_list])
    if len(list_types) != 1:
        raise ValueError("List is not type homogeneous. " f"Existing types: {list_types}")


def flatten_dict(
    input_dict: Dict[str, JsonTypesUnion],
    key_separator: str = " ",
    keys_to_remove: List = ["metrics"],
) -> Dict:
    output_dict = dict()
    # iteration with inner recursion
    for key, value in input_dict.items():
        if isinstance(value, dict):
            flat_inner_dict = flatten_dict(value)
            for inner_key, inner_value in flat_inner_dict.items():
                new_key = (
                    key + key_separator + inner_key
                    if key not in keys_to_remove
                    else inner_key
                )
                output_dict[new_key] = inner_value
        else:
            # keys to remove are not applied for lowest level keys
            output_dict[key] = value
    return output_dict


def flatten_list(input_list: List, ensure_type_homogeneity: bool = False) -> List:
    output_list = list()
    # iteration with inner recursion
    for value in input_list:
        if isinstance(value, list):
            inner_flat_list = flatten_list(value)
            for inner_value in inner_flat_list:
                output_list.append(inner_value)
        else:
            output_list.append(value)
    if ensure_type_homogeneity:
        ensure_list_types_homogeneity(output_list)
    return output_list


def get_module_members(
    module_names_chain: Union[List, str]
) -> Tuple[ModuleContentMap, ModuleContentMap]:
    def get_module_name(module_names_chain: List[str]) -> str:
        name = module_names_chain[0]
        for subname in module_names_chain[1:]:
            name += "." + subname
        return name

    def merge_maps(
        first_map: ModuleContentMap, second_map: ModuleContentMap
    ) -> ModuleContentMap:
        output = dict()
        all_keys = set(first_map.keys()) | set(second_map.keys())
        for key in all_keys:
            if key in first_map and key in second_map:
                output[key] = first_map[key] + second_map[key]
            elif key in first_map:
                output[key] = first_map[key]
            elif key in second_map:
                output[key] = second_map[key]
        return output

    if isinstance(module_names_chain, str):
        module_names_chain = [module_names_chain]
    module_name = get_module_name(module_names_chain)
    classes_map: ModuleContentMap = dict()
    functions_map: ModuleContentMap = dict()

    try:
        module = importlib.__import__(module_name, globals(), locals(), [], 0)
        for subname in module_names_chain[1:]:
            module = getattr(module, subname)
    except ModuleNotFoundError:
        return dict(), dict()

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if name in classes_map and obj not in classes_map[name]:
                classes_map[name].append(obj)
            else:
                classes_map[name] = [obj]
        elif inspect.isfunction(obj):
            if name in functions_map and obj not in functions_map[name]:
                functions_map[name].append(obj)
            else:
                functions_map[name] = [obj]

    if hasattr(module, "__all__"):
        for name in module.__all__:
            sub_classes_map, sub_functions_map = get_module_members(
                module_names_chain + [name]
            )
            classes_map = merge_maps(classes_map, sub_classes_map)
            functions_map = merge_maps(functions_map, sub_functions_map)

    return classes_map, functions_map


def is_float(value: str) -> bool:
    return (
        re.match(
            r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$", value
        )
        is not None
    )


def convert_to_numeric_if_possible(value: str) -> Union[Numeric, str]:
    if value.isdigit():
        return int(value)
    elif is_float(value):
        return float(value)
    else:
        return value


def convert_to_numpy(a, dp_compat=False) -> np.ndarray:
    if dp_compat and ("dpctl" in str(type(a)) or "dpnp" in str(type(a))):
        return a
    if isinstance(a, np.ndarray):
        return a
    elif hasattr(a, "to_numpy"):
        return a.to_numpy()
    elif hasattr(a, "asnumpy"):
        return a.asnumpy()
    elif "dpnp" in str(type(a)):
        import dpnp

        return dpnp.asnumpy(a)
    elif "dpctl" in str(type(a)):
        import dpctl.tensor

        return dpctl.tensor.to_numpy(a)
    elif "cupy.ndarray" in str(type(a)):
        return a.get()
    else:
        raise ValueError("Unable to convert data to numpy.ndarray")
