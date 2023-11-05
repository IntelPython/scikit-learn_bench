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
import datetime
import shutil
import subprocess
from typing import Any, Dict, List, Union

import utils
from pathlib import Path
import hashlib


def get_configs(path: Path) -> List[str]:
    result = list()
    for dir_or_file in os.listdir(path):
        new_path = Path(path, dir_or_file)
        if dir_or_file.endswith('.json'):
            result.append(str(new_path))
        elif os.path.isdir(new_path):
            result += get_configs(new_path)
    return result

allowed_analysis_types = ['vtune', 'emon', 'psrecord', 'ittpy']

def check_additional_orders(args, common_params):
    result = {}
    if args.sgx_gramine:
        result['sgx_gramine'] = args.sgx_gramine

    analysis_config = {}
    if 'analysis' in common_params.keys():
        for analyse_type in allowed_analysis_types:
            if analyse_type in common_params['analysis'].keys():
                analysis_config[analyse_type] = common_params['analysis'][analyse_type]

    result.update(analysis_config)
    return result

def get_program_name(analysis_config):
    program_name = 'python'
    if 'sgx_gramine' in analysis_config.keys() and analysis_config['sgx_gramine'] == True:
        program_name = 'gramine-sgx ./sklearnex'
    return program_name

def dict_to_cmd_args(dictionary):
    results = []
    for key, item in dictionary.items():
        if isinstance(item, list):
            for subitem in item:
                results.append(f'{key} {subitem}')
        else:
            results.append(f'{key} {item}')
    return " ".join(results)

def get_analyse_prefix(analysis_config):
    for key in allowed_analysis_types:
        if key in analysis_config.keys():
            args = dict_to_cmd_args(analysis_config[key])
            return args
    else:
        return None

def get_benchmark_extra_args(analysis_config):
    result = []
    for key in analysis_config.keys():
        result.append(f'--{key}'.replace('_', '-'))
    return ' '.join(result)

emon_dat_file_name, emon_xlsx_file_name = None, None

def vtune_postproc(analysis_config, analysis_folder):
    vtune_foldername = os.path.join(analysis_folder, 'vtune_'+analysis_folder.split('/')[-1])
    if '-r' in analysis_config['vtune'].keys():
        shutil.move(analysis_config['vtune']['-r'], vtune_foldername)
    else:
        for file in os.listdir('.'):
            if 'r00' in file:
                shutil.move(file, vtune_foldername)

def fetch_expected_emon_filename(edp_config_path):
    with open(edp_config_path, 'r') as edp_config:
        for line in edp_config:
            if line.strip().startswith('EMON_DATA='):
                emon_dat_file_name = line.strip()[10:]
            if line.strip().startswith('OUTPUT='):
                emon_xlsx_file_name = line.strip()[7:]
        else:
            return 'emon.dat', 'summary.xlsx'
        return emon_dat_filename, emon_xlsx_file_name

def emon_postproc(analysis_config, analysis_folder):
    global emon_dat_file_name, emon_xlsx_file_name
    if emon_dat_file_name == None:
        emon_dat_file_name, emon_xlsx_file_name = fetch_expected_emon_filename('utils/emon/edp_config.txt')
    if '-f' in analysis_config['emon'].keys():
        shutil.move(analysis_config['emon']['-f'], emon_dat_file_name)
    else:
        shutil.move('emon.dat', emon_dat_file_name)
    emon_processing_command = 'emon -process-edp ./utils/emon/edp_config.txt'
    res = subprocess.run(emon_processing_command.split(' '), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, encoding='utf-8')
    if res.stderr[:-1] != '':
        logging.error(f'EMON error message: {res.stderr[:-1]}')
    shutil.move(emon_dat_file_name, analysis_folder)
    shutil.move(emon_xlsx_file_name, analysis_folder)

def psrecord_postproc(analysis_config, analysis_folder):
    if '--log' in analysis_config.keys():
        shutil.move(analysis_config['--log'], analysis_folder)
    if '--plot' in analysis_config.keys():
        shutil.move(analysis_config['--plot'], analysis_folder)

def postproc_analysis_result(analysis_config, analysis_folder):
    if 'vtune' in analysis_config.keys():
        vtune_postproc(analysis_config, analysis_folder)
    elif 'emon' in analysis_config.keys():
        emon_postproc(analysis_config, analysis_folder)
    elif 'psrecord' in analysis_config.keys():
        psrecord_postproc(analysis_config, analysis_folder)

def emon_preproc(analysis_config, command_line):
    emon_sh = 'emon_runner.sh'
    subcommand = f'#!/bin/bash\n' \
        f'{command_line}\n'
    with open(emon_sh, 'w') as f:
        f.write(subcommand)
    os.chmod(emon_sh, 0o755)
    return emon_sh, subcommand

def preproc_analysis(analysis_config, analysis_prefix, bench_command_line):
    subcommand = ''
    if 'emon' in analysis_config.keys():
        emon_sh, subcommand = emon_preproc(analysis_config, bench_command_line)
        command = f'emon {analysis_prefix} ./{emon_sh}'
    elif 'psrecord' in analysis_config.keys():
        command = f'psrecord {analysis_prefix} "{bench_command_line}"'
    elif 'vtune' in analysis_config.keys():
        command = f'vtune {analysis_prefix} -- {bench_command_line}'
    else:
        command = bench_command_line
    return command, subcommand

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
    parser.add_argument('--sgx-gramine', default=False, action='store_true',
                        help='Run benchmarks with Gramine & Intel(R) SGX.')
    parser.add_argument('--vtune', default=False, action='store_true',
                        help='Profile benchmarks with VTune.')
    parser.add_argument('--emon', default=False, action='store_true',
                        help='Profile benchmarks with EMON.')
    parser.add_argument('--psrecord', default=False, action='store_true',
                        help='Analyze memory consumption with psrecord')
    parser.add_argument('--box-filter-measurements-analysis', default=500, type=int,
                        help='Maximum number of measurements in box filter (analysed stage). '
                        'When benchmark uses this parameter to understand number of '
                        'runs for target stage, other stage will be run '
                        'only for getting trained model.'
                        'Parameter won\'t be used if analysis options aren\'t enabled.')
    parser.add_argument('--time-limit-analysis', default=100., type=float,
                        help='Time to spend to currently analysed stage. '
                        'When benchmark uses this parameter to calculate '
                        'time for target stage, other stage will be run '
                        'only for getting trained model.'
                        'Parameter won\'t be used if analysis options aren\'t enabled.')
    parser.add_argument('--box-filter-measurements', type=int, default=100,
                        help='Maximum number of measurements in box filter.')
    parser.add_argument('--time-limit', default=10., type=float,
                        help='Target time to spend to benchmark.')
    parser.add_argument('--flush-caches', default=False,
                        action='store_true',
                        help='Should benchmark flush CPU caches after each run during measuring.'
                        'Recommended for default runs and vtune profiling (in case you would like to flush caches).')
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
    timestamp = str(datetime.date.today()).replace(' ', '--').replace(':', '-').replace('.', '-')

    logging.basicConfig(
        stream=sys.stdout, format='%(levelname)s: %(message)s', level=args.verbose)
    hostname = socket.gethostname()

    env = os.environ.copy()
    if 'DATASETSROOT' in env:
        datasets_root = env['DATASETSROOT']
        logging.info(f'Datasets folder at {datasets_root}')
    elif 'DAAL_DATASETS' in env:
        datasets_root = env['DAAL_DATASETS']
        logging.info(f'Datasets folder at {datasets_root}')
    else:
        datasets_root = ''
        logging.info('Datasets folder is not set, using local folder')

    json_result: Dict[str, Union[Dict[str, Any], List[Any]]] = {
        'common': {},
        'hardware': utils.get_hw_parameters(),
        'software': utils.get_sw_parameters(),
        'results': []
    }
    json_result['common']['timestamp'] = timestamp

    path_to_analysis_dir = 'analysis_'+timestamp
    if os.path.exists(path_to_analysis_dir):
        shutil.rmtree(path_to_analysis_dir)
    os.makedirs(path_to_analysis_dir)

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
            # print('PRE PARAMS:', params)
            analysis_config = check_additional_orders(args, common_params)
            # print('ANALYSIS CONFIG:', analysis_config)
            if 'analysis' in params.keys():
                del params['analysis']
            # print('POST PARAMS:', params)
            program_name = get_program_name(analysis_config)
            analysis_prefix = get_analyse_prefix(analysis_config)
            bench_extra_args = get_benchmark_extra_args(analysis_config)
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

            if (len(libs) * len(cases) == 0):
                continue

            for dataset in params_set['dataset']:
                if dataset['source'] in ['csv', 'npy']:
                    dataset_name = dataset['name'] if 'name' in dataset else 'unknown'
                    if 'training' not in dataset or 'x' not in dataset['training']:
                        logging.warning(
                            f'Dataset {dataset_name} could not be loaded. \n'
                            'Training data for algorithm is not specified'
                            )
                        continue

                    files = {}

                    files['file-X-train'] = dataset['training']["x"]
                    if 'y' in dataset['training']:
                        files['file-y-train'] = dataset['training']["y"]
                    if 'testing' in dataset:
                        files['file-X-test'] = dataset["testing"]["x"]
                        if 'y' in dataset['testing']:
                            files['file-y-test'] = dataset["testing"]["y"]

                    dataset_path = utils.find_the_dataset(dataset_name, datasets_root,
                                                          files.values())
                    if dataset_path is None:
                        logging.warning(
                            f'Dataset {dataset_name} could not be loaded. \n'
                            'Check the correct name or expand the download in '
                            'the folder dataset.'
                            )
                        continue
                    elif not dataset_path and datasets_root:
                        logging.info(
                            f'{dataset_name} is taken from local folder'
                            )

                    paths = ''
                    for data_path, data_file in files.items():
                        paths += f'--{data_path} {os.path.join(dataset_path, data_file)} '

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

                    files = {}
                    gen_args.filex = f'{file_prefix}X-train{file_postfix}'
                    files['file-X-train'] = gen_args.filex
                    if gen_args.type not in ['blobs']:
                        gen_args.filey = f'{file_prefix}y-train{file_postfix}'
                        files['file-y-train'] = gen_args.filey

                    if 'testing' in dataset:
                        gen_args.test_samples = dataset['testing']['n_samples']
                        gen_args.filextest = f'{file_prefix}X-test{file_postfix}'
                        files['file-X-test'] = gen_args.filextest
                        if gen_args.type not in ['blobs']:
                            gen_args.fileytest = f'{file_prefix}y-test{file_postfix}'
                            files['file-y-test'] = gen_args.fileytest
                    else:
                        gen_args.test_samples = 0
                        gen_args.filextest = gen_args.filex
                        files['file-X-test'] = gen_args.filextest
                        if gen_args.type not in ['blobs']:
                            gen_args.fileytest = gen_args.filey
                            files['file-y-test'] = gen_args.filey

                    dataset_name = f'synthetic_{gen_args.type}'

                    dataset_path = utils.find_or_gen_dataset(gen_args,
                                                             datasets_root, files.values())
                    if dataset_path is None:
                        logging.warning(
                            f'Dataset {dataset_name} could not be generated. \n'
                        )
                        continue

                    paths = ''
                    for data_path, data_file in files.items():
                        paths += f'--{data_path} {os.path.join(dataset_path, data_file)} '
                else:
                    logging.warning('Unknown dataset source. Only synthetics datasets '
                                    'and csv/npy files are supported now')

                no_intel_optimize = \
                    '--no-intel-optimized ' if args.no_intel_optimized else ''
                for lib in libs:
                    for i, case in enumerate(cases):
                        analysis_stage_collection = ['default']
                        if analysis_prefix != None:
                            analysis_stage_collection.extend(['fit', 'infer'])
                        for analysis_stage in analysis_stage_collection:
                            bench_command_line = f'{program_name} {lib}_bench/{algorithm}.py ' \
                                + no_intel_optimize \
                                + f'--arch {hostname} {case} {paths} ' \
                                + f'--dataset-name {dataset_name} ' \
                                + f'--box-filter-measurements-analysis {args.box_filter_measurements_analysis} ' \
                                + f'--box-filter-measurements {args.box_filter_measurements} ' \
                                + f'--time-limit-analysis {args.time_limit_analysis} ' \
                                + f'--time-limit {args.time_limit} ' \
                                + f'--target-stage {analysis_stage} '
                            if args.flush_caches:
                                bench_command_line += ' --flush-caches '
                            hash_of_case = hashlib.sha256(bench_command_line.encode('utf-8')).hexdigest()
                            if analysis_stage == 'default':
                                command = bench_command_line
                                subcommand = None
                            else:
                                bench_command_line += f' {bench_extra_args} '
                                command, subcommand = preproc_analysis(analysis_config, analysis_prefix, bench_command_line)

                            command = ' '.join(command.split())

                            logging.info(command)
                            if 'emon' in analysis_config.keys() and subcommand != None:
                                logging.info(f'Subcommand: {subcommand}')
                            if not args.dummy_run:
                                case_result = f'{lib},{algorithm} ' + case
                                stdout, stderr = utils.read_output_from_command(command, env=os.environ.copy())
                                stdout, extra_stdout = utils.filter_stdout(stdout)
                                stderr = utils.filter_stderr(stderr)
                                try:
                                    output_json = json.loads(stdout)
                                    json_decoding_ok = True
                                except json.JSONDecodeError as decoding_exception:
                                    stderr += f'CASE {case_result} JSON DECODING ERROR:\n' \
                                        + f'{decoding_exception}\n{stdout}\n'
                                    json_decoding_ok = False
                                if analysis_stage != 'default':
                                    actual_config = None
                                    for cfg in output_json:
                                        if cfg['stage'] == 'training' and analysis_stage == 'fit':
                                            actual_config = cfg
                                        elif cfg['stage'] != 'training' and analysis_stage != 'fit':
                                            actual_config = cfg
                                    current_rows_number = actual_config['input_data']['rows']
                                    current_columns_number = actual_config['input_data']['columns']
                                    case_folder = f"{lib}_{algorithm}_{analysis_stage}_{dataset_name}_{current_rows_number}x{current_columns_number}_{hash_of_case[:6]}"
                                    analysis_folder = os.path.join(path_to_analysis_dir, case_folder)
                                    os.makedirs(analysis_folder)
                                    postproc_analysis_result(analysis_config, analysis_folder)

                                if analysis_prefix != None:
                                    for item in output_json:
                                        item['hash_prefix'] = hash_of_case[:6]
                                        item['analysis'] = analysis_config

                                if json_decoding_ok and analysis_stage == 'default':
                                    json_result['results'].extend(output_json)

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
