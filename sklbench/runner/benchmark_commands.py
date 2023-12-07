# ===============================================================================
# Copyright 2023 Intel Corporation
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
import sys
from time import time
from typing import Dict, List, Tuple

from ..utils.bench_case import get_bench_case_name, get_bench_case_value
from ..utils.common import custom_format, hash_from_json_repr, read_output_from_command
from ..utils.custom_types import BenchCase
from ..utils.logger import logger


def generate_benchmark_command(
    bench_case: BenchCase, filters: List[BenchCase], log_level: str
):
    # generate parameter and filter arguments for benchmark cli wrapper
    bench_case_str = json.dumps(bench_case).replace(" ", "")
    filters_str = json.dumps({"filters": filters}).replace(" ", "")
    # get command prefix if set
    command_prefix = ""
    taskset = get_bench_case_value(bench_case, "bench:taskset")
    vtune_profiling = get_bench_case_value(bench_case, "bench:vtune_profiling")
    if vtune_profiling is not None:
        if sys.platform == "linux":
            vtune_result_dir = get_bench_case_value(
                bench_case, "bench:vtune_results_directory", "vtune_results"
            )
            os.makedirs(vtune_result_dir, exist_ok=True)
            vtune_result_path = os.path.join(
                vtune_result_dir,
                "_".join(
                    [
                        get_bench_case_name(bench_case, shortened=True, separator="_"),
                        hash_from_json_repr(bench_case),
                        # TODO: replace unix time in ms with datetime
                        str(int(time() * 1000)),
                    ]
                ),
            )
            command_prefix = (
                f"{command_prefix}vtune -collect {vtune_profiling} "
                f"-r {vtune_result_path} -start-paused -q -no-summary "
            )
            # vtune CLI requires modification of quotes bench args: `"` -> `\"`
            bench_case_str = bench_case_str.replace('"', '\\"')
            filters_str = filters_str.replace('"', '\\"')
        else:
            logger.warning(
                "Intel(R) VTune(TM) profiling in scikit-learn_bench "
                "is supported only on Linux."
            )
    if taskset is not None:
        command_prefix = f"{command_prefix}taskset -c {taskset} "
    # benchmark selection
    if get_bench_case_value(bench_case, "algorithm:estimator") is not None:
        benchmark_name = "sklearn_estimator"
    elif get_bench_case_value(bench_case, "algorithm:function") is not None:
        benchmark_name = "custom_function"
    else:
        raise ValueError("Unknown benchmark type")
    return (
        f"{command_prefix}python "
        f"-m sklbench.benchmarks.{benchmark_name} "
        f"--bench-case {bench_case_str} "
        f"--filters {filters_str} "
        f"--log-level {log_level}"
    )


def run_benchmark_from_case(
    bench_case: BenchCase, filters: List[BenchCase], log_level: str
) -> Tuple[int, List[Dict]]:
    command = generate_benchmark_command(bench_case, filters, log_level)
    logger.debug(f"Benchmark wrapper call command: {command}")
    return_code, stdout, stderr = read_output_from_command(command)

    if stdout != "":
        logger.debug(f'{custom_format("Benchmark stdout:", bcolor="OKBLUE")}\n{stdout}')
    if return_code == 0:
        if stderr != "":
            logger.warning(f"Benchmark stderr:\n{stderr}")
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("Unable to read benchmark output in json format.")
            return_code = -1
            result = list()
    else:
        logger.warning(f"Benchmark returned non-zero code={return_code}.")
        logger.warning(f"Benchmark stderr:\n{stderr}")
        result = list()
    return return_code, result
