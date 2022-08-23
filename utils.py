# ===============================================================================
# Copyright 2020-2021 Intel Corporation
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
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, cast

from datasets.make_datasets import try_gen_dataset
from datasets.load_datasets import try_load_dataset


def filter_stderr(text: str) -> str:
    # delete 'Intel(R) Extension for Scikit-learn usage in sklearn' messages
    fake_error_message = ('Intel(R) Extension for Scikit-learn* enabled ' +
                          '(https://github.com/intel/scikit-learn-intelex)')

    return ''.join(text.split(fake_error_message))


def filter_stdout(text: str) -> Tuple[str, str]:
    verbosity_letters = 'EWIDT'
    filtered, extra = '', ''
    for line in text.split('\n'):
        if line == '':
            continue
        to_remove = False
        for letter in verbosity_letters:
            if line.startswith(f'[{letter}]'):
                to_remove = True
                break
        if to_remove:
            extra += line + '\n'
        else:
            filtered += line + '\n'
    return filtered, extra


def files_in_folder(folder: str, files: Iterable[str]) -> bool:
    for file in files:
        if not os.path.isfile(os.path.join(folder, file)):
            return False
    return True


def find_or_gen_dataset(args: Any, folder: str, files: Iterable[str]):
    if files_in_folder("", files):
        return ""
    if folder:
        if files_in_folder(folder, files) or \
           try_gen_dataset(args, folder):
            return folder
    if try_gen_dataset(args, ""):
        return ""
    return None


def find_the_dataset(name: str, folder: str, files: Iterable[str]):
    if files_in_folder("", files):
        return ""
    if folder:
        if files_in_folder(folder, files) or \
           try_load_dataset(dataset_name=name,
                            output_directory=Path(os.path.join(folder, "data"))):
            return folder
    if try_load_dataset(dataset_name=name, output_directory=Path("data")):
        return ""
    return None


def read_output_from_command(command: str,
                             env: Dict[str, str] = os.environ.copy()) -> Tuple[str, str]:
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] += ":" + os.path.dirname(os.path.abspath(__file__))
    else:
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', env=env)
    return res.stdout[:-1], res.stderr[:-1]


def parse_lscpu_lscl_info(command_output: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    for elem in command_output.strip().split('\n'):
        splt = elem.split(':')
        res[splt[0]] = splt[1]
    return res


def get_hw_parameters() -> Dict[str, Union[Dict[str, Any], float]]:
    if 'Linux' not in platform.platform():
        return {}

    hw_params: Dict[str, Union[Dict[str, str], float]] = {'CPU': {}}
    # get CPU information
    lscpu_info, _ = read_output_from_command('lscpu')
    lscpu_info = ' '.join(lscpu_info.split())
    for line in lscpu_info.split('\n'):
        k, v = line.split(": ")[:2]
        if k == 'CPU MHz':
            continue
        cast(Dict[str, str], hw_params['CPU'])[k] = v

    # get RAM size
    mem_info, _ = read_output_from_command('free -b')
    mem_info = mem_info.split('\n')[1]
    mem_info = ' '.join(mem_info.split())
    hw_params['RAM size[GB]'] = int(mem_info.split(' ')[1]) / 2 ** 30

    # get Intel GPU information
    try:
        lsgpu_info, _ = read_output_from_command(
            'lscl --device-type=gpu --platform-vendor=Intel')
        device_num = 0
        start_idx = lsgpu_info.find('Device ')
        while start_idx >= 0:
            start_idx = lsgpu_info.find(':', start_idx) + 1
            end_idx = lsgpu_info.find('Device ', start_idx)
            hw_params[f'GPU Intel #{device_num + 1}'] = parse_lscpu_lscl_info(
                lsgpu_info[start_idx: end_idx])
            device_num += 1
            start_idx = end_idx
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # get Nvidia GPU information
    try:
        gpu_info, _ = read_output_from_command(
            'nvidia-smi --query-gpu=name,memory.total,driver_version,pstate '
            '--format=csv,noheader')
        gpu_info_arr = gpu_info.split(', ')
        if len(gpu_info_arr) == 0:
            return hw_params
        hw_params['GPU Nvidia'] = {
            'Name': gpu_info_arr[0],
            'Memory size': gpu_info_arr[1],
            'Performance mode': gpu_info_arr[3]
        }
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        pass
    return hw_params


def get_sw_parameters() -> Dict[str, Dict[str, Any]]:
    sw_params = {}
    try:
        gpu_info, _ = read_output_from_command(
            'nvidia-smi --query-gpu=name,memory.total,driver_version,pstate '
            '--format=csv,noheader')
        info_arr = gpu_info.split(', ')
        sw_params['GPU_driver'] = {'version': info_arr[2]}
        # alert if GPU is already running any processes
        gpu_processes, _ = read_output_from_command(
            'nvidia-smi --query-compute-apps=name,pid,used_memory '
            '--format=csv,noheader')
        if gpu_processes != '':
            print(f'There are running processes on GPU:\n{gpu_processes}',
                  file=sys.stderr)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass

    # get python packages info from conda
    try:
        conda_list, _ = read_output_from_command('conda list --json')
        needed_columns = ['version', 'build_string', 'channel']
        conda_list_json: List[Dict[str, str]] = json.loads(conda_list)
        for pkg in conda_list_json:
            pkg_info = {}
            for col in needed_columns:
                if col in pkg:
                    pkg_info[col] = pkg[col]
            sw_params[pkg['name']] = pkg_info
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass

    return sw_params


def generate_cases(params: Dict[str, Union[List[Any], Any]]) -> List[str]:
    '''
    Generate cases for benchmarking by iterating the parameter values
    '''
    commands = ['']
    for param, values in params.items():
        if isinstance(values, list):
            prev_len = len(commands)
            commands *= len(values)
            dashes = '-' if len(param) == 1 else '--'
            for command_num in range(prev_len):
                for idx, val in enumerate(values):
                    commands[prev_len * idx + command_num] += ' ' + \
                        dashes + param + ' ' + str(val)
        else:
            dashes = '-' if len(param) == 1 else '--'
            for command_num, _ in enumerate(commands):
                commands[command_num] += ' ' + \
                    dashes + param + ' ' + str(values)
    return commands
