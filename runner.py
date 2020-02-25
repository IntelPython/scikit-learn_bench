# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
import os
import sys
import subprocess
import json
from platform import platform
from make_datasets import (
    gen_regression, gen_classification, gen_kmeans, gen_blobs
)


def verbose_print(text):
    global verbose_mode
    if verbose_mode:
        print(text)


def filter_stderr(text):
    # delete 'Intel(R) DAAL usage in sklearn' messages
    fake_error_message = 'Intel(R) Data Analytics Acceleration Library ' \
                         + '(Intel(R) DAAL) solvers for sklearn enabled: ' \
                         + 'https://intelpython.github.io/daal4py/sklearn.html'
    while fake_error_message in text:
        text = text.replace(fake_error_message, '')
    return text


def generate_cases(params):
    '''
    Generate cases for benchmarking by iterating of
    parameters values
    '''
    global cases
    if len(params) == 0:
        return cases
    prev_length = len(cases)
    param_name = list(params.keys())[0]
    n_param_values = len(params[param_name])
    cases = cases * n_param_values
    dashes = '-' if len(param_name) == 1 else '--'
    for i in range(n_param_values):
        for j in range(prev_length):
            cases[prev_length * i + j] += f' {dashes}{param_name} ' \
                                          + f'{params[param_name][i]}'
    del params[param_name]
    generate_cases(params)


def read_output_from_command(command):
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8')
    return res.stdout[:-1], res.stderr[:-1]


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='ConfigPath',
                    type=argparse.FileType('r'), default='config_example.json',
                    help='Path to configuration file')
parser.add_argument('--dummy-run', default=False, action='store_true',
                    help='Run configuration parser and datasets generation'
                         'without benchmarks running')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Print additional information during'
                         'benchmarks running')
parser.add_argument('--output-format', default='json', choices=('json', 'csv'),
                    help='Output type of benchmarks to use with their runner')
args = parser.parse_args()
verbose_mode = args.verbose

with open(args.config.name, 'r') as config_file:
    config = json.load(config_file)

# make directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)

csv_result = ''
json_result = {'hardware': {}, 'results': []}
if 'Linux' in platform():
    hostname, _ = read_output_from_command('hostname')
    batch, _ = read_output_from_command('date -Iseconds')
    # get CPU information
    lscpu_info, _ = read_output_from_command('lscpu')
    # remove excess spaces in CPU info output
    while '  ' in lscpu_info:
        lscpu_info = lscpu_info.replace('  ', ' ')
    lscpu_info = lscpu_info.split('\n')
    for i in range(len(lscpu_info)):
        lscpu_info[i] = lscpu_info[i].split(': ')
    json_result['hardware'].update(
        {'CPU': {line[0]: line[1] for line in lscpu_info}})

# get parameters that are common for all cases
common_params = config['common']
for params_set in config['cases']:
    cases = ['']
    params = common_params.copy()
    params.update(params_set.copy())
    algorithm = params['algorithm']
    libs = params['lib']
    del params['dataset'], params['algorithm'], params['lib']
    generate_cases(params)
    verbose_print(f'{algorithm} algorithm: {len(libs) * len(cases)} case(s),'
                  + f' {len(params_set["dataset"])} dataset(s)\n')
    for dataset in params_set['dataset']:
        if dataset['training'].startswith('synth'):
            class GenerationArgs:
                pass
            gen_args = GenerationArgs()
            paths = ''

            if 'seed' in params_set.keys():
                gen_args.seed = params_set['seed']
            else:
                gen_args.seed = 777

            dataset_params = dataset['training'].split('_')
            gen_args.task = dataset_params[1]
            gen_args.samples = int(dataset_params[2])
            gen_args.features = int(dataset_params[3])
            if gen_args.task in ['clsf', 'kmeans', 'blobs']:
                cls_num_for_file = '-' + dataset_params[4]
                gen_args.classes = int(dataset_params[4])
                gen_args.clusters = gen_args.classes
            else:
                cls_num_for_file = ''

            file_prefix = f'data/synth-{gen_args.task}{cls_num_for_file}-'
            file_postfix = f'-{gen_args.samples}x{gen_args.features}.npy'

            if gen_args.task == 'kmeans':
                gen_args.node_id = 0
                gen_args.filei = f'{file_prefix}init{file_postfix}'
                paths += f'--filei {gen_args.filei}'
                gen_args.filet = f'{file_prefix}threshold{file_postfix}'

            gen_args.filex = f'{file_prefix}X-train{file_postfix}'
            paths += f' --file-X-train {gen_args.filex}'
            if gen_args.task not in ['kmeans', 'blobs']:
                gen_args.filey = f'{file_prefix}y-train{file_postfix}'
                paths += f' --file-y-train {gen_args.filey}'

            if 'testing' in dataset.keys():
                dataset_params = dataset['testing'].split('_')
                _, _, gen_args.test_samples, _ = dataset_params[:4]
                gen_args.filextest = f'{file_prefix}X-test{file_postfix}'
                paths += f' --file-X-test {gen_args.filextest}'
                if gen_args.task not in ['kmeans', 'blobs']:
                    gen_args.fileytest = f'{file_prefix}y-test{file_postfix}'
                    paths += f' --file-y-test {gen_args.fileytest}'
            else:
                gen_args.test_samples = 0
                gen_args.filextest = gen_args.filex
                if gen_args.task not in ['kmeans', 'blobs']:
                    gen_args.fileytest = gen_args.filey

            if not args.dummy_run and not os.path.isfile(gen_args.filex):
                if gen_args.task == 'reg':
                    gen_regression(gen_args)
                elif gen_args.task == 'clsf':
                    gen_classification(gen_args)
                elif gen_args.task == 'kmeans':
                    gen_kmeans(gen_args)
                elif gen_args.task == 'blobs':
                    gen_blobs(gen_args)
        else:
            raise ValueError(
                'Unknown dataset. Only synthetics are supported now')
        for lib in libs:
            for i, case in enumerate(cases):
                command = f'python {lib}/{algorithm}.py --batch {batch} ' \
                          + f'--arch {hostname} --header --output-format ' \
                          + f'{args.output_format}{case} {paths}'
                while '  ' in command:
                    command = command.replace('  ', ' ')
                verbose_print(command)
                if not args.dummy_run:
                    stdout, stderr = read_output_from_command(command)
                    stderr = filter_stderr(stderr)
                    if args.output_format == 'json':
                        json_result['results'] += json.loads(f'[{stdout}]')
                    elif args.output_format == 'csv':
                        csv_result += stdout + '\n'
                    if stderr != '':
                        print(stderr, file=sys.stderr)

if args.output_format == 'json':
    json_result = json.dumps(json_result, indent=4)
    print(json_result, end='\n')
elif args.output_format == 'csv':
    print(csv_result, end='')
