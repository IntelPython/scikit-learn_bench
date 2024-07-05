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
from typing import Dict

from ..utils.bench_case import get_bench_case_value, get_data_name
from ..utils.custom_types import BenchCase
from ..utils.logger import logger


def enrich_result(result: Dict, bench_case: BenchCase) -> Dict:
    """Common function for all benchmarks to update
    the result with additional information"""
    result.update(
        {
            "dataset": get_data_name(bench_case, shortened=True),
            "library": get_bench_case_value(bench_case, "algorithm:library").replace(
                "sklbench.emulators.", ""
            ),
            "device": get_bench_case_value(bench_case, "algorithm:device"),
        }
    )
    enable_modelbuilders = get_bench_case_value(
        bench_case, "algorithm:enable_modelbuilders", False
    )
    if enable_modelbuilders and result["library"] in ["xgboost", "lightgbm", "catboost"]:
        # NOTE: while modelbuilders are stored in `daal4py.mb` namespace
        # their results are saved as `sklearnex` for better report readability
        logger.debug(
            "Modelbuilders are enabled, changing library "
            f"`{result['library']}` to `sklearnex` in benchmark output."
        )
        result["library"] = "sklearnex"
    taskset = get_bench_case_value(bench_case, "bench:taskset", None)
    if taskset is not None:
        result.update({"taskset": taskset})
    distributor = get_bench_case_value(bench_case, "bench:distributor")
    if distributor is not None:
        result.update({"distributor": distributor})
    mpi_params = get_bench_case_value(bench_case, "bench:mpi_params", dict())
    for mpi_key, mpi_value in mpi_params.items():
        result[f"mpi_{mpi_key}"] = mpi_value
    return result


def check_to_print_result(bench_case: BenchCase) -> bool:
    """Check if the benchmark should print the result"""
    distribution = get_bench_case_value(bench_case, "bench:distributor")
    if distribution == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank != 0:
            return False
    return True


def main_template(main_method):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-case", required=True, type=str)
    parser.add_argument("--filters", required=True, type=str)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for benchmark",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    bench_case = json.loads(args.bench_case)
    filters = json.loads(args.filters)["filters"]

    results = main_method(bench_case, filters)

    if check_to_print_result(bench_case):
        print(json.dumps(results, indent=4))
