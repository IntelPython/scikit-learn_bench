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

import timeit

import numpy as np

from .bench_case import get_bench_case_value
from .custom_types import BenchCase
from .logger import logger

try:
    import itt

    itt_is_available = True
except (ImportError, ModuleNotFoundError):
    itt_is_available = False


def box_filter(timing, left=0.2, right=0.8):
    timing.sort()
    size = len(timing)
    if size == 1:
        return timing[0] * 1000, 0
    lower, upper = timing[int(size * left)], timing[int(size * right)]
    result = np.array([item for item in timing if lower < item < upper])
    return np.mean(result) * 1000, np.std(result) * 1000


def measure_time(
    func,
    *args,
    n_runs=20,
    time_limit=60 * 60,
    std_mean_ratio=0.2,
    enable_itt=False,
    **kwargs,
):
    if enable_itt and not itt_is_available:
        logger.warning(
            "Intel(R) VTune(TM) profiling was requested "
            'but "itt" python module is not available.'
        )
    times = []
    func_return_value = None
    while len(times) < n_runs:
        if enable_itt and itt_is_available:
            itt.resume()
        t0 = timeit.default_timer()
        func_return_value = func(*args, **kwargs)
        t1 = timeit.default_timer()
        if enable_itt and itt_is_available:
            itt.pause()
        times.append(t1 - t0)
        if sum(times) > time_limit:
            logger.warning(
                f"'{func}' function measurement time "
                f"({sum(times)} seconds from {len(times)} runs) "
                f"exceeded time limit ({time_limit} seconds)"
            )
            break
    mean, std = box_filter(times)
    if std / mean > std_mean_ratio:
        logger.warning(
            f'Measured "std / mean" time ratio of "{str(func)}" function is higher '
            f"than threshold ({round(std / mean, 3)} vs. {std_mean_ratio})"
        )
    return mean, std, func_return_value


# wrapper to get measurement params from benchmarking case
def measure_case(case: BenchCase, func, *args, **kwargs):
    distirbutor = get_bench_case_value(case, "bench:distributor")
    if distirbutor == "mpi":
        # sync all MPI processes
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        comm.Barrier()
    return measure_time(
        func,
        *args,
        **kwargs,
        n_runs=get_bench_case_value(case, "bench:n_runs", 10),
        time_limit=get_bench_case_value(case, "bench:time_limit", 3600),
        enable_itt=get_bench_case_value(case, "bench:vtune_profiling") is not None,
    )
