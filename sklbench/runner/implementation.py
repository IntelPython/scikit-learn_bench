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
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

from psutil import cpu_count
from tqdm import tqdm

from ..datasets import load_data
from ..report import generate_report, get_result_tables_as_df
from ..utils.bench_case import get_bench_case_name, get_data_name
from ..utils.common import custom_format, hash_from_json_repr
from ..utils.config import early_filtering, generate_bench_cases, generate_bench_filters
from ..utils.custom_types import BenchCase
from ..utils.env import get_environment_info
from ..utils.logger import logger
from .commands_helper import run_benchmark_from_case


def call_benchmarks(
    bench_cases: List[BenchCase],
    filters: List[BenchCase],
    log_level: str = "WARNING",
    environment_name: Union[str, None] = None,
    early_exit: bool = False,
) -> Tuple[int, Dict[str, Union[Dict, List]]]:
    """Iterates over benchmarking cases with progress bar and combines their results"""
    env_info = get_environment_info()
    if environment_name is None:
        environment_name = hash_from_json_repr(env_info)
    results = list()
    return_code = 0
    bench_cases_with_pbar = tqdm(bench_cases)
    for bench_case in bench_cases_with_pbar:
        bench_cases_with_pbar.set_description(
            custom_format(
                get_bench_case_name(bench_case, shortened=True), bcolor="HEADER"
            )
        )
        try:
            bench_return_code, bench_entries = run_benchmark_from_case(
                bench_case, filters, log_level
            )
            if bench_return_code != 0:
                return_code = bench_return_code
                if early_exit:
                    break
            for entry in bench_entries:
                entry["environment_name"] = environment_name
                results.append(entry)
        except KeyboardInterrupt:
            return_code = -1
            break
    full_result = {
        "bench_cases": results,
        "environment": {environment_name: env_info},
    }
    return return_code, full_result


def run_benchmarks(args: argparse.Namespace) -> int:
    # overwrite all logging levels if requested
    if args.log_level is not None:
        for log_type in ["runner", "bench", "report"]:
            setattr(args, f"{log_type}_log_level", args.log_level)
    # set logging level
    logger.setLevel(args.runner_log_level)

    # find and parse configs
    bench_cases = generate_bench_cases(args)

    # get parameter filters
    param_filters = generate_bench_filters(args.parameter_filters)

    # perform early filtering based on 'data' parameters and
    # some of 'algorithm' parameters assuming they were already assigned
    bench_cases = early_filtering(bench_cases, param_filters)

    # prefetch datasets
    if args.prefetch_datasets:
        # trick: get unique dataset names only to avoid loading of same dataset
        # by different cases/processes
        dataset_cases = {get_data_name(case): case for case in bench_cases}
        logger.debug(f"Unique dataset names to load:\n{list(dataset_cases.keys())}")
        n_proc = min([16, cpu_count(), len(dataset_cases)])
        logger.info(f"Prefetching datasets with {n_proc} processes")
        with Pool(n_proc) as pool:
            pool.map(load_data, dataset_cases.values())

    # run bench_cases
    return_code, result = call_benchmarks(
        bench_cases,
        param_filters,
        args.bench_log_level,
        args.environment_name,
        args.exit_on_error,
    )

    # output as pandas dataframe
    if len(result["bench_cases"]) != 0:
        for key, df in get_result_tables_as_df(result).items():
            logger.info(f'{custom_format(key, bcolor="HEADER")}\n{df}')

    # output raw result
    logger.debug(custom_format(result))

    with open(args.result_file, "w") as fp:
        json.dump(result, fp, indent=4)

    # generate report
    if args.report:
        if args.result_file not in args.result_files:
            args.result_files += [args.result_file]
        generate_report(args)

    return return_code
