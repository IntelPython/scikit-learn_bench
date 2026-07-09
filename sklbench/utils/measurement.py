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

import gc
import threading
import timeit
import warnings
from math import ceil, sqrt
from time import sleep
from typing import Dict, List

import numpy as np
import psutil
from cpuinfo import get_cpu_info
from scipy.stats import pearsonr

from .bench_case import get_bench_case_value
from .custom_types import BenchCase, BenchResult
from .env import get_number_of_sockets
from .logger import logger

try:
    import itt

    itt_is_available = True
except (ImportError, ModuleNotFoundError):
    itt_is_available = False

try:
    import pynvml

    pynvml.nvmlInit()

    nvml_is_available = True
except (ImportError, ModuleNotFoundError):
    nvml_is_available = False


def large_scale_measurements(timing):
    first_iter = timing[0] * 1000
    mean = np.mean(timing[1:]) * 1000
    stdev = np.std(timing[1:]) * 1000
    timing_sorted = np.sort(timing)
    Q1, Q3 = np.percentile(timing_sorted, [25, 75])
    IQ = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQ, Q3 + 1.5 * IQ

    filtered_times = timing_sorted[(timing_sorted >= lower) & (timing_sorted <= upper)]

    box_filter_mean = np.mean(filtered_times) * 1000 if filtered_times.size > 0 else 0
    box_filter_stdev = np.std(filtered_times) * 1000 if filtered_times.size > 0 else 0
    return mean, stdev, first_iter, box_filter_mean, box_filter_stdev


def box_filter(array, left=0.2, right=0.8):
    array.sort()
    size = len(array)
    if size == 1 or len(np.unique(array)) == 1:
        return array[0], 0.0
    lower, upper = array[int(size * left)], array[int(size * right)]
    result = np.array([item for item in array if lower < item < upper])
    return np.mean(result), np.std(result)


def enrich_metrics(
    bench_result: BenchResult, include_performance_stability_metrics=False
):
    """Transforms raw performance and other results into aggregated metrics"""
    # time metrics
    res = bench_result.copy()
    if isinstance(res["time[ms]"], list):
        mean, std = box_filter(res["time[ms]"])
        if include_performance_stability_metrics:
            res.update(
                {
                    "1st run time[ms]": res["time[ms]"][0],
                    "1st-mean run ratio": res["time[ms]"][0] / mean,
                }
            )
        res.update(
            {
                "time[ms]": mean,
                "time CV": std / mean,  # Coefficient of Variation
            }
        )
    else:
        # already aggregated (e.g. from a baseline file)
        mean = res["time[ms]"]
        std = res.get("time std[ms]", 0.0)
        if mean != 0:
            res["time CV"] = std / mean
        else:
            res["time CV"] = 0.0
    cost = res.get("cost[microdollar]", None)
    if cost and isinstance(cost, list):
        res["cost[microdollar]"] = box_filter(res["cost[microdollar]"])[0]
    batch_size = res.get("batch_size", None)
    if batch_size:
        res["throughput[samples/ms]"] = (
            (res["samples"] // batch_size) * batch_size
        ) / mean
    # memory metrics
    for memory_type in ["RAM", "VRAM"]:
        if f"peak {memory_type} usage[MB]" in res:
            if include_performance_stability_metrics:
                with warnings.catch_warnings():
                    # ignoring ConstantInputWarning
                    warnings.filterwarnings(
                        "ignore",
                        message="An input array is constant; the correlation coefficient is not defined",
                    )
                    mem_iter_corr, _ = pearsonr(
                        res[f"peak {memory_type} usage[MB]"],
                        list(range(len(res[f"peak {memory_type} usage[MB]"]))),
                    )
                res[f"{memory_type} usage-iteration correlation"] = mem_iter_corr
            res[f"peak {memory_type} usage[MB]"] = max(
                res[f"peak {memory_type} usage[MB]"]
            )
    # cpu metrics
    if "cpu load[%]" in res:
        res["cpu load[%]"] = np.median(res["cpu load[%]"])
    return res


def get_n_from_cache_size():
    """Gets `n` size of square matrix that fits into L3 cache"""
    cache_size = 0
    cpu_info = get_cpu_info()
    # cache reading abibility of cpuinfo is platform dependent
    if "l3_cache_size" in cpu_info:
        cache_size += cpu_info["l3_cache_size"]
    if "l2_cache_size" in cpu_info:
        cache_size += cpu_info["l2_cache_size"] * psutil.cpu_count(logical=False)
    n_sockets = get_number_of_sockets()
    return ceil(sqrt(n_sockets * cache_size / 8))


def flush_cache(n: int | None = None):
    if n is None:
        n = get_n_from_cache_size()
    np.matmul(np.random.rand(n, n), np.random.rand(n, n))


def get_ram_usage():
    """Memory used by the current process in bytes"""
    return psutil.Process().memory_info().rss


def get_vram_usage():
    """Memory used by the current process on all GPUs in bytes"""
    pid = psutil.Process().pid

    device_count = pynvml.nvmlDeviceGetCount()
    vram_usage = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in process_info:
            if p.pid == pid:
                vram_usage += p.usedGpuMemory
    return vram_usage


def monitor_memory_usage(
    interval: float, memory_profiles: Dict[str, List], stop_event, enable_nvml_profiling
):
    while not stop_event.is_set():
        memory_profiles["RAM"].append(get_ram_usage())
        if enable_nvml_profiling:
            memory_profiles["VRAM"].append(get_vram_usage())
        sleep(interval)

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
    inners, iters = [], []
    while len(times) < n_runs:
        if enable_itt and itt_is_available:
            itt.resume()
        t0 = timeit.default_timer()
        func_return_value = func(*args, **kwargs)
        t1 = timeit.default_timer()
        if hasattr(func, "__self__") and hasattr(func.__self__, "_n_inner_iter"):
            inners.append(func.__self__._n_inner_iter)
            iters.append(func.__self__.n_iter_)
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

    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.debug(
                "iters across n runs: "
                + str(iters)
                + ", inner iters across n runs: "
                + str(inners)
            )
            logger.debug(f"Runtime for all {n_runs} iterations: {times}")
    except ModuleNotFoundError:
        logger.debug(f"Runtime for all {n_runs} iterations: {times}")
    # mean, std = box_filter(times)
    # if std / mean > std_mean_ratio:
    #    logger.warning(
    #        f'Measured "std / mean" time ratio of "{str(func)}" function is higher '
    #        f"than threshold ({round(std / mean, 3)} vs. {std_mean_ratio})"
    #    )
    return large_scale_measurements(times)

def measure_perf(
    func,
    *args,
    n_runs: int,
    time_limit: float,
    enable_itt: bool,
    collect_return_values: bool = False,
    enable_cache_flushing: bool,
    enable_garbage_collection: bool,
    enable_cpu_profiling: bool,
    enable_memory_profiling: bool,
    enable_nvml_profiling: bool = False,
    memory_profiling_interval: float = 0.001,
    cost_per_hour: float = 0.0,
    **kwargs,
):
    if enable_itt and not itt_is_available:
        logger.warning(
            "Intel(R) VTune(TM) profiling was requested "
            'but "itt" python module is not available.'
        )
        enable_itt = False
    times = list()
    if collect_return_values:
        func_return_values = list()
    if enable_cpu_profiling:
        cpu_loads = list()
    if enable_memory_profiling:
        memory_peaks = {"RAM": list()}
        if enable_nvml_profiling:
            memory_peaks["VRAM"] = list()
    while len(times) < n_runs:
        if enable_cache_flushing:
            flush_cache()
        if enable_itt:
            itt.resume()
        if enable_memory_profiling:
            memory_profiles = {"RAM": list()}
            if enable_nvml_profiling:
                memory_profiles["VRAM"] = list()
            profiling_stop_event = threading.Event()
            profiling_thread = threading.Thread(
                target=monitor_memory_usage,
                args=(
                    memory_profiling_interval,
                    memory_profiles,
                    profiling_stop_event,
                    enable_nvml_profiling,
                ),
            )
            profiling_thread.start()
        if enable_cpu_profiling:
            # start cpu profiling interval by using `None` value
            psutil.cpu_percent(interval=None)
        t0 = timeit.default_timer()
        func_return_value = func(*args, **kwargs)
        t1 = timeit.default_timer()
        if enable_cpu_profiling:
            cpu_loads.append(psutil.cpu_percent(interval=None))
        if enable_memory_profiling:
            profiling_stop_event.set()
            profiling_thread.join()
            memory_peaks["RAM"].append(max(memory_profiles["RAM"]))
            if enable_nvml_profiling:
                memory_peaks["VRAM"].append(max(memory_profiles["VRAM"]))
        if collect_return_values:
            func_return_values.append(func_return_value)
        if enable_itt:
            itt.pause()
        times.append((t1 - t0))
        if enable_garbage_collection:
            gc.collect()
        if sum(times) > time_limit:
            logger.warning(
                f"'{func}' function measurement time "
                f"({sum(times)} seconds from {len(times)} runs) "
                f"exceeded time limit ({time_limit} seconds)"
            )
            break
    perf_metrics = {"time[ms]": list(map(lambda x: x * 1000, times))}
    if enable_memory_profiling:
        perf_metrics[f"peak RAM usage[MB]"] = list(
            map(lambda x: x / 2**20, memory_peaks["RAM"])
        )
        if enable_nvml_profiling:
            perf_metrics[f"peak VRAM usage[MB]"] = list(
                map(lambda x: x / 2**20, memory_peaks["VRAM"])
            )
    if enable_cpu_profiling:
        perf_metrics["cpu load[%]"] = cpu_loads
    if cost_per_hour > 0.0:
        perf_metrics["cost[microdollar]"] = list(
            map(lambda x: x / 1000 / 3600 * cost_per_hour * 1e6, perf_metrics["time[ms]"])
        )
    if collect_return_values:
        return perf_metrics, func_return_values
    else:
        return perf_metrics


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
