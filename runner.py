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

import argparse
import json
import logging
import os
import socket
import sys
from typing import Any, Dict, List, Union

import datasets.make_datasets as make_datasets
import utils
from pathlib import Path


def get_configs(path: Path) -> List[str]:
    result = list()
    for dir_or_file in os.listdir(path):
        new_path = Path(path, dir_or_file)
        if dir_or_file.endswith('.json'):
            result.append(str(new_path))
        elif os.path.isdir(new_path):
            result += get_configs(new_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', metavar='ConfigPath', type=str,
                        default='configs/config_example.json',
                        help='The path to a configuration file or '
                             'a directory that contains configuration files')
    parser.add_argument('--device', '--devices', default='host cpu gpu none', type=str, nargs='+',
                        choices=('host', 'cpu', 'gpu', 'none'),
                        help='Availible execution context devices. '
                        'This parameter only marks devices as available, '
                        'make sure to add the device to the config file '
                        'to run it on a specific device')
    parser.add_argument('--dummy-run', default=False, action='store_true',
                        help='Run configuration parser and datasets generation '
                             'without benchmarks running')
    parser.add_argument('--dtype', '--dtypes', type=str, default="float32 float64", nargs='+',
                        choices=("float32", "float64"),
                        help='Available floating point data types'
                        'This parameter only marks dtype as available, '
                        'make sure to add the dtype parameter to the config file ')
    parser.add_argument('--workload-size', type=str, default="small medium large", nargs='+',
                        choices=("small", "medium", "large"),
                        help='Available workload sizes,'
                        'make sure to add the workload-size parameter to the config file '
                        'unmarked workloads will be launched anyway')
    parser.add_argument('--no-intel-optimized', default=False, action='store_true',
                        help='Use Scikit-learn without Intel optimizations')
    parser.add_argument('--output-file', default='results.json',
                        type=argparse.FileType('w'),
                        help='Output file of benchmarks to use with their runner')
    parser.add_argument('--verbose', default='INFO', type=str,
                        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help='Print additional information during benchmarks running')
    parser.add_argument('--report', nargs='?', default=None, metavar='ConfigPath', type=str,
                        const='report_generator/default_report_gen_config.json',
                        help='Create an Excel report based on benchmarks results. '
                        'If the parameter is not set, the reporter will not be launched. '
                        'If the parameter is set and the config is not specified, '
                        'the default config will be used. '
                        'Need "openpyxl" library')
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout, format='%(levelname)s: %(message)s', level=args.verbose)
    hostname = socket.gethostname()

    # make directory for data if it doesn't exist
    os.makedirs('data', exist_ok=True)

    json_result: Dict[str, Union[Dict[str, Any], List[Any]]] = {
        'hardware': utils.get_hw_parameters(),
        'software': utils.get_sw_parameters(),
        'results': []
    }
    is_successful = True
    # getting jsons from folders
    paths_to_configs: List[str] = list()
    for config_name in args.configs.split(','):
        if os.path.isdir(config_name):
            config_name = get_configs(Path(config_name))
        else:
            config_name = [config_name]
        paths_to_configs += config_name
    args.configs = ','.join(paths_to_configs)

    for config_name in args.configs.split(','):
        logging.info(f'Config: {config_name}')
        with open(config_name, 'r') as config_file:
            config = json.load(config_file)

        # get parameters that are common for all cases
        common_params = config['common']
        for params_set in config['cases']:
            params = common_params.copy()
            params.update(params_set.copy())

            if 'workload-size' in params:
                if params['workload-size'] not in args.workload_size:
                    continue
                del params['workload-size']

            device = []
            if 'device' not in params:
                if 'sklearn' in params['lib']:
                    logging.info('The device parameter value is not defined in config, '
                                 'none is used')
                device = ['none']
            elif not isinstance(params['device'], list):
                device = [params['device']]
            else:
                device = params['device']
            params["device"] = [dv for dv in device if dv in args.device]

            dtype = []
            if 'dtype' not in params:
                dtype = ['float64']
            elif not isinstance(params['dtype'], list):
                dtype = [params['dtype']]
            else:
                dtype = params['dtype']
            params['dtype'] = [dt for dt in dtype if dt in args.dtype]

            algorithm = params['algorithm']
            libs = params['lib']
            if not isinstance(libs, list):
                libs = [libs]
            del params['dataset'], params['algorithm'], params['lib']
            cases = utils.generate_cases(params)
            logging.info(f'{algorithm} algorithm: {len(libs) * len(cases)} case(s),'
                         f' {len(params_set["dataset"])} dataset(s)\n')

            for dataset in params_set['dataset']:
                if dataset['source'] in ['csv', 'npy']:
                    dataset_name = dataset['name'] if 'name' in dataset else 'unknown'
                    if 'training' not in dataset or \
                        'x' not in dataset['training'] or \
                        not utils.find_the_dataset(dataset_name,
                                                   dataset['training']['x']):
                        logging.warning(
                            f'Dataset {dataset_name} could not be loaded. \n'
                            'Check the correct name or expand the download in '
                            'the folder dataset.')
                        continue
                    paths = '--file-X-train ' + dataset['training']["x"]
                    if 'y' in dataset['training']:
                        paths += ' --file-y-train ' + dataset['training']["y"]
                    if 'testing' in dataset:
                        paths += ' --file-X-test ' + dataset["testing"]["x"]
                        if 'y' in dataset['testing']:
                            paths += ' --file-y-test ' + \
                                dataset["testing"]["y"]
                elif dataset['source'] == 'synthetic':
                    class GenerationArgs:
                        classes: int
                        clusters: int
                        features: int
                        filex: str
                        filextest: str
                        filey: str
                        fileytest: str
                        samples: int
                        seed: int
                        test_samples: int
                        type: str
                    gen_args = GenerationArgs()
                    paths = ''

                    if 'seed' in params_set:
                        gen_args.seed = params_set['seed']
                    else:
                        gen_args.seed = 777

                    # default values
                    gen_args.clusters = 10
                    gen_args.type = dataset['type']
                    gen_args.samples = dataset['training']['n_samples']
                    gen_args.features = dataset['n_features']
                    if 'n_classes' in dataset:
                        gen_args.classes = dataset['n_classes']
                        cls_num_for_file = f'-{dataset["n_classes"]}'
                    elif 'n_clusters' in dataset:
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

                    if 'testing' in dataset:
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

                no_intel_optimize = \
                    '--no-intel-optimized ' if args.no_intel_optimized else ''
                for lib in libs:
                    for i, case in enumerate(cases):
                        command = f'python {lib}_bench/{algorithm}.py ' \
                            + no_intel_optimize \
                            + f'--arch {hostname} {case} {paths} ' \
                            + f'--dataset-name {dataset_name}'
                        command = ' '.join(command.split())
                        logging.info(command)
                        if not args.dummy_run:
                            case = f'{lib},{algorithm} ' + case
                            stdout, stderr = utils.read_output_from_command(
                                command, env=os.environ.copy())
                            stdout, extra_stdout = utils.filter_stdout(stdout)
                            stderr = utils.filter_stderr(stderr)

                            print(stdout, end='\n')

                            if extra_stdout != '':
                                stderr += f'CASE {case} EXTRA OUTPUT:\n' \
                                    + f'{extra_stdout}\n'
                            try:
                                if isinstance(json_result['results'], list):
                                    json_result['results'].extend(
                                        json.loads(stdout))
                            except json.JSONDecodeError as decoding_exception:
                                stderr += f'CASE {case} JSON DECODING ERROR:\n' \
                                    + f'{decoding_exception}\n{stdout}\n'

                            if stderr != '':
                                if 'daal4py' not in stderr:
                                    is_successful = False
                                    logging.warning(
                                        'Error in benchmark: \n' + stderr)

    json.dump(json_result, args.output_file, indent=4)
    name_result_file = args.output_file.name
    args.output_file.close()

    if args.report:
        command = 'python report_generator/report_generator.py ' \
            + f'--result-files {name_result_file} '              \
            + f'--report-file {name_result_file}.xlsx '          \
            + '--generation-config ' + args.report
        logging.info(command)
        stdout, stderr = utils.read_output_from_command(command)
        if stderr != '':
            logging.warning('Error in report generator: \n' + stderr)
            is_successful = False

    if not is_successful:
        logging.warning('benchmark running had runtime errors')
        sys.exit(1)
