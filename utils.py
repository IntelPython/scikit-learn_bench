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
import logging
import multiprocessing
import os
import platform
import subprocess
import sys


def filter_stderr(text):
    # delete 'Intel(R) Extension for Scikit-learn usage in sklearn' messages
    fake_error_message = 'Intel(R) Extension for Scikit-learn* enabled ' + \
                         '(https://github.com/intel/scikit-learn-intelex)'
    while fake_error_message in text:
        text = text.replace(fake_error_message, '')
    return text


def filter_stdout(text):
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


def is_exists_files(files):
    for f in files:
        if not os.path.isfile(f):
            return False
    return True


def read_output_from_command(command, env=os.environ.copy()):
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] += ":" + os.path.dirname(os.path.abspath(__file__))
    else:
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', env=env)
    return res.stdout[:-1], res.stderr[:-1]


def _is_ht_enabled():
    try:
        cpu_info, _ = read_output_from_command('lscpu')
        cpu_info = cpu_info.split('\n')
        for el in cpu_info:
            if 'Thread(s) per core' in el:
                threads_per_core = int(el[-1])
                if threads_per_core > 1:
                    return True
                else:
                    return False
        return False
    except FileNotFoundError:
        logging.info('Impossible to check hyperthreading via lscpu')
        return False


def get_omp_env():
    cpu_count = multiprocessing.cpu_count()
    omp_num_threads = str(cpu_count // 2) if _is_ht_enabled() else str(cpu_count)

    omp_env = {
        'OMP_PLACES': f'{{0}}:{cpu_count}:1',
        'OMP_NUM_THREADS': omp_num_threads
    }
    return omp_env


def parse_lscpu_lscl_info(command_output):
    command_output = command_output.strip().split('\n')
    for i in range(len(command_output)):
        command_output[i] = command_output[i].split(':')
    return {line[0].strip(): line[1].strip() for line in command_output}


def get_hw_parameters():
    hw_params = {}

    if 'Linux' in platform.platform():
        # get CPU information
        lscpu_info, _ = read_output_from_command('lscpu')
        hw_params.update({'CPU': parse_lscpu_lscl_info(lscpu_info)})
        if 'CPU MHz' in hw_params['CPU'].keys():
            del hw_params['CPU']['CPU MHz']

        # get RAM size
        mem_info, _ = read_output_from_command('free -b')
        mem_info = mem_info.split('\n')[1]
        while '  ' in mem_info:
            mem_info = mem_info.replace('  ', ' ')
        mem_info = int(mem_info.split(' ')[1]) / 2 ** 30
        hw_params.update({'RAM size[GB]': mem_info})

        # get Intel GPU information
        try:
            lsgpu_info, _ = read_output_from_command(
                'lscl --device-type=gpu --platform-vendor=Intel')
            platform_num = 0
            start_idx = lsgpu_info.find('Platform ')
            while start_idx >= 0:
                start_idx = lsgpu_info.find(':', start_idx) + 1
                end_idx = lsgpu_info.find('Platform ', start_idx)
                platform_info = parse_lscpu_lscl_info(lsgpu_info[start_idx:end_idx])
                hw_params.update({f'GPU Intel platform {platform_num + 1}': platform_info})
                platform_num += 1
                start_idx = end_idx
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # get Nvidia GPU information
        try:
            gpu_info, _ = read_output_from_command(
                'nvidia-smi --query-gpu=name,memory.total,driver_version,pstate '
                '--format=csv,noheader')
            gpu_info = gpu_info.split(', ')
            hw_params.update({
                'GPU Nvidia': {
                    'Name': gpu_info[0],
                    'Memory size': gpu_info[1],
                    'Performance mode': gpu_info[3]
                }
            })
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return hw_params


def get_sw_parameters():
    sw_params = {}
    try:
        gpu_info, _ = read_output_from_command(
            'nvidia-smi --query-gpu=name,memory.total,driver_version,pstate '
            '--format=csv,noheader')
        gpu_info = gpu_info.split(', ')

        sw_params.update(
            {'GPU_driver': {'version': gpu_info[2]}})
        # alert if GPU is already running any processes
        gpu_processes, _ = read_output_from_command(
            'nvidia-smi --query-compute-apps=name,pid,used_memory '
            '--format=csv,noheader')
        if gpu_processes != '':
            print(f'There are running processes on GPU:\n{gpu_processes}',
                  file=sys.stderr)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # get python packages info from conda
    try:
        conda_list, _ = read_output_from_command('conda list --json')
        needed_columns = ['version', 'build_string', 'channel']
        conda_list = json.loads(conda_list)
        for pkg in conda_list:
            pkg_info = {}
            for col in needed_columns:
                if col in pkg.keys():
                    pkg_info.update({col: pkg[col]})
            sw_params.update({pkg['name']: pkg_info})
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return sw_params
