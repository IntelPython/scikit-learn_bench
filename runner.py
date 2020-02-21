# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
import os
import sys
import subprocess
import json
from platform import platform
from make_datasets import gen_regression, gen_classification, gen_kmeans


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
    for i in range(n_param_values):
        for j in range(prev_length):
            cases[prev_length * i + j] += ' {}{} {}'.format(
                '-' if len(param_name) == 1 else '--',
                param_name, params[param_name][i])
    del params[param_name]
    generate_cases(params)


read_stdout_from_command = lambda command: subprocess.run(
    command.split(' '), stdout=subprocess.PIPE, encoding='utf-8').stdout[:-1]

parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='ConfigPath',
                    type=argparse.FileType('r'), default='config_example.json',
                    help='Path to configuration file')
parser.add_argument('--dummy-run', default=False, action='store_true',
                    help='Run configuration parser and datasets generation'
                         'without benchmarks running')
parser.add_argument('--output-format', default='json', choices=('json', 'csv'),
                    help='Output type of benchmarks to use with their runner')
args = parser.parse_args()

with open(args.config.name, 'r') as config_file:
    config = json.load(config_file)

# make directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)

csv_result = ''
json_result = {'hardware': {}, 'results':[]}
if 'Linux' in platform():
    hostname = read_stdout_from_command('hostname')
    batch = read_stdout_from_command('date -Iseconds')
    # get CPU information (only on Linux)
    lscpu_info = read_stdout_from_command('lscpu')
    # remove excess spaces in CPU info output
    while '  ' in lscpu_info:
        lscpu_info = lscpu_info.replace('  ',' ')
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
    print('\n{} algorithm: {} case(s), {} dataset(s)\n'.format(
        algorithm, len(cases), len(params_set['dataset'])))
    for dataset in params_set['dataset']:
        if dataset['training'].startswith('synth'):
            class GenerationArgs: pass
            gen_args = GenerationArgs()
            paths = ''

            if 'seed' in params_set.keys():
                gen_args.seed = params_set['seed']
            else:
                gen_args.seed = 777

            dataset_params = dataset['training'].split('_')
            _, gen_args.task, gen_args.samples, gen_args.features = dataset_params[:4]
            gen_args.samples = int(gen_args.samples)
            gen_args.features = int(gen_args.features)
            if gen_args.task in ['clsf', 'clust']:
                cls_num_for_file = '-' + dataset_params[4]
                gen_args.classes = int(dataset_params[4])
                if gen_args.task == 'clust':
                    gen_args.clusters = int(dataset_params[4])
                    gen_args.node_id = 0
                    gen_args.filei = f'data/synth-clust-{gen_args.clusters}-init-{gen_args.samples}x{gen_args.features}.npy'
                    paths += f'--filei {gen_args.filei}'
                    gen_args.filet = f'data/synth-clust-{gen_args.clusters}-threshold-{gen_args.samples}x{gen_args.features}.npy'
            else:
                cls_num_for_file = ''

            gen_args.filex = f'data/synth-{gen_args.task}{cls_num_for_file}-X-train-{gen_args.samples}x{gen_args.features}.npy'
            paths += f' --file-X-train {gen_args.filex}'
            if gen_args.task != 'clust':
                    gen_args.filey = f'data/synth-{gen_args.task}{cls_num_for_file}-y-train-{gen_args.samples}x{gen_args.features}.npy'
                    paths += f' --file-y-train {gen_args.filey}'

            if 'testing' in dataset.keys():
                dataset_params = dataset['testing'].split('_')
                _, _, gen_args.test_samples, _ = dataset_params[:4]
                gen_args.filextest = f'data/synth-{gen_args.task}{cls_num_for_file}-X-test-{gen_args.test_samples}x{gen_args.features}.npy'
                paths += f' --file-X-test {gen_args.filextest}'
                if gen_args.task != 'clust':
                    gen_args.fileytest = f'data/synth-{gen_args.task}{cls_num_for_file}-y-test-{gen_args.test_samples}x{gen_args.features}.npy'
                    paths += f' --file-y-test {gen_args.fileytest}'
            else:
                gen_args.test_samples = 0
                gen_args.filextest = gen_args.filex
                if gen_args.task != 'clust':
                    gen_args.fileytest = gen_args.filey

            if not args.dummy_run and not os.path.isfile(gen_args.filex):
                if gen_args.task == 'reg':
                    gen_regression(gen_args)
                elif gen_args.task == 'clsf':
                    gen_classification(gen_args)
                elif gen_args.task == 'clust':
                    gen_kmeans(gen_args)
        else:
            raise ValueError(
                'Unknown dataset. Only synthetics are supported now')
        for lib in libs:
            for i, case in enumerate(cases):
                command = f'python {lib}/{algorithm}.py --batch {batch} --arch {hostname} --output-format {args.output_format}{case} {paths}'
                while '  ' in command:
                    command = command.replace('  ', ' ')
                print(command)
                if not args.dummy_run:
                    r = subprocess.run(
                        command.split(' '), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, encoding='utf-8')
                    if args.output_format == 'json':
                        json_result['results'] += json.loads(f'[{r.stdout}]')
                    elif args.output_format == 'csv':
                        csv_result += r.stdout
                    print(r.stderr, file=sys.stderr)

if args.output_format == 'json':
    json_result = json.dumps(json_result, indent=4)
    print(json_result, end='\n')
elif args.output_format == 'csv':
    print(csv_result)
