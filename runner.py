#===============================================================================
# Copyright 2020 Intel Corporation
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
#===============================================================================

import argparse
import os
import sys
import json
import socket
import logging
import pathlib

import datasets.make_datasets as make_datasets
import utils

from datasets.load_datasets import try_load_dataset


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', metavar='ConfigPath', type=str,
                        default='configs/config_example.json',
                        help='Path to configuration files')
    parser.add_argument('--dummy-run', default=False, action='store_true',
                        help='Run configuration parser and datasets generation'
                             'without benchmarks running')
    parser.add_argument('--no-intel-optimized', default=False, action='store_true',
                        help='Use no intel optimized version. '
                             'Now avalible for scikit-learn benchmarks'),
    parser.add_argument('--output-file', default='results.json',
                        type=argparse.FileType('w'),
                        help='Output file of benchmarks to use with their runner')
    parser.add_argument('--verbose', default='INFO', type=str,
                        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help='Print additional information during benchmarks running')
    parser.add_argument('--report', default=False, action='store_true',
                        help='Create an Excel report based on benchmarks results. '
                             'Need "openpyxl" library')
    args = parser.parse_args()
    env = os.environ.copy()

    logging.basicConfig(
        stream=sys.stdout, format='%(levelname)s: %(message)s', level=args.verbose)
    hostname = socket.gethostname()

    # make directory for data if it doesn't exist
    os.makedirs('data', exist_ok=True)

    json_result = {'hardware': {}, 'software': {}, 'results': []}
    is_successful = True

    for config_name in args.configs.split(','):
        logging.info(f'Config: {config_name}')
        with open(config_name, 'r') as config_file:
            config = json.load(config_file)

        if 'omp_env' not in config.keys():
            config['omp_env'] = []
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
            logging.info(f'{algorithm} algorithm: {len(libs) * len(cases)} case(s),'
                         f' {len(params_set["dataset"])} dataset(s)\n')

            for dataset in params_set['dataset']:
                if dataset['source'] in ['csv', 'npy']:
                    train_data = dataset["training"]
                    test_data = dataset["testing"]

                    file_train_data_x = train_data["x"]
                    file_train_data_y = train_data["y"]
                    file_test_data_x = test_data["x"]
                    file_test_data_y = test_data["y"]
                    paths = f'--file-X-train {file_train_data_x}'
                    if 'y' in dataset['training'].keys():
                        paths += f' --file-y-train {file_train_data_y}'
                    if 'testing' in dataset.keys():
                        paths += f' --file-X-test {file_test_data_x}'
                        if 'y' in dataset['testing'].keys():
                            paths += f' --file-y-test {file_test_data_y}'
                    if 'name' in dataset.keys():
                        dataset_name = dataset['name']
                    else:
                        dataset_name = 'unknown'

                    if not utils.is_exists_files([file_train_data_x, file_train_data_y]):
                        directory_dataset = pathlib.Path(file_train_data_x).parent
                        if not try_load_dataset(dataset_name=dataset_name,
                                                output_directory=directory_dataset):
                            logging.warning(f'Dataset {dataset_name} '
                                            'could not be loaded. \n'
                                            'Check the correct name or expand '
                                            'the download in the folder dataset.')
                            continue

                elif dataset['source'] == 'synthetic':
                    class GenerationArgs:
                        pass
                    gen_args = GenerationArgs()
                    paths = ''

                    if 'seed' in params_set.keys():
                        gen_args.seed = params_set['seed']
                    else:
                        gen_args.seed = 777

                    # default values
                    gen_args.clusters = 10
                    gen_args.type = dataset['type']
                    gen_args.samples = dataset['training']['n_samples']
                    gen_args.features = dataset['n_features']
                    if 'n_classes' in dataset.keys():
                        gen_args.classes = dataset['n_classes']
                        cls_num_for_file = f'-{dataset["n_classes"]}'
                    elif 'n_clusters' in dataset.keys():
                        gen_args.clusters = dataset['n_clusters']
                        cls_num_for_file = f'-{dataset["n_clusters"]}'
                    else:
                        cls_num_for_file = ''

                    file_prefix = f'data/synthetic-{gen_args.type}{cls_num_for_file}-'
                    file_postfix = f'-{gen_args.samples}x{gen_args.features}.npy'

                    gen_args.filex = f'{file_prefix}X-train{file_postfix}'
                    paths += f' --file-X-train {gen_args.filex}'
                    if gen_args.type not in ['blobs']:
                        gen_args.filey = f'{file_prefix}y-train{file_postfix}'
                        paths += f' --file-y-train {gen_args.filey}'

                    if 'testing' in dataset.keys():
                        gen_args.test_samples = dataset['testing']['n_samples']
                        gen_args.filextest = f'{file_prefix}X-test{file_postfix}'
                        paths += f' --file-X-test {gen_args.filextest}'
                        if gen_args.type not in ['blobs']:
                            gen_args.fileytest = f'{file_prefix}y-test{file_postfix}'
                            paths += f' --file-y-test {gen_args.fileytest}'
                    else:
                        gen_args.test_samples = 0
                        gen_args.filextest = gen_args.filex
                        if gen_args.type not in ['blobs']:
                            gen_args.fileytest = gen_args.filey

                    if not args.dummy_run and not os.path.isfile(gen_args.filex):
                        if gen_args.type == 'regression':
                            make_datasets.gen_regression(gen_args)
                        elif gen_args.type == 'classification':
                            make_datasets.gen_classification(gen_args)
                        elif gen_args.type == 'blobs':
                            make_datasets.gen_blobs(gen_args)
                    dataset_name = f'synthetic_{gen_args.type}'
                else:
                    logging.warning('Unknown dataset source. Only synthetics datasets '
                                    'and csv/npy files are supported now')

                omp_env = utils.get_omp_env()
                no_intel_optimize = \
                    '--no-intel-optimized ' if args.no_intel_optimized else ''
                for lib in libs:
                    env = os.environ.copy()
                    if lib == 'xgboost':
                        for var in config['omp_env']:
                            env[var] = omp_env[var]
                    for i, case in enumerate(cases):
                        command = f'python {lib}_bench/{algorithm}.py ' \
                            + no_intel_optimize \
                            + f'--arch {hostname} {case} {paths} ' \
                            + f'--dataset-name {dataset_name}'
                        while '  ' in command:
                            command = command.replace('  ', ' ')
                        logging.info(command)
                        if not args.dummy_run:
                            case = f'{lib},{algorithm} ' + case
                            stdout, stderr = utils.read_output_from_command(
                                command)
                            stdout, extra_stdout = utils.filter_stdout(stdout)
                            stderr = utils.filter_stderr(stderr)

                            print(stdout, end='\n')

                            if extra_stdout != '':
                                stderr += f'CASE {case} EXTRA OUTPUT:\n' \
                                    + f'{extra_stdout}\n'
                            try:
                                json_result['results'].extend(
                                    json.loads(stdout))
                            except json.JSONDecodeError as decoding_exception:
                                stderr += f'CASE {case} JSON DECODING ERROR:\n' \
                                    + f'{decoding_exception}\n{stdout}\n'
                            if stderr != '':
                                is_successful = False
                                logging.warning('Error in benchmark: \n' + stderr)

    json.dump(json_result, args.output_file, indent=4)
    name_result_file = args.output_file.name
    args.output_file.close()

    if args.report:
        command = 'python report_generator/report_generator.py ' \
            + f'--result-files {name_result_file} '              \
            + f'--report-file {name_result_file}.xlsx '          \
            + '--generation-config report_generator/default_report_gen_config.json'
        logging.info(command)
        stdout, stderr = utils.read_output_from_command(command)
        if stderr != '':
            logging.warning('Error in report generator: \n' + stderr)
            is_successful = False

    if not is_successful:
        logging.warning('benchmark running had runtime errors')
        sys.exit(1)
