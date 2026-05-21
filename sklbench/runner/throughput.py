# ===============================================================================
# Copyright 2026 Intel Corporation
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
import subprocess
import time
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..utils.barrier import accept_and_wait, create_server, send_all, wait_all
from ..utils.bench_case import get_bench_case_name, get_bench_case_value
from ..utils.common import custom_format, hash_from_json_repr
from ..utils.core_assignment import compute_core_assignments
from ..utils.custom_types import BenchCase
from ..utils.env import get_environment_info, get_numa_cpus_conf
from ..utils.logger import logger


def validate_throughput_args(
    num_instances: int, cores_per_instance: int, measurement_duration: float
):
    if num_instances is None or num_instances < 1:
        raise ValueError(
            "bench:num_instances is required and must be >= 1 in throughput mode"
        )
    if cores_per_instance is None or cores_per_instance < 1:
        raise ValueError(
            "bench:cores_per_instance is required and must be >= 1 in throughput mode"
        )
    if measurement_duration <= 0:
        raise ValueError("bench:measurement_duration must be > 0")


def validate_sync_quality(instance_outputs: List[Dict], stage: str):
    """Check if all instances started measurement within 100ms of each other."""
    start_times = []
    for output in instance_outputs:
        if stage in output.get("stages", {}):
            timestamps = output["stages"][stage].get("start_ts", [])
            if timestamps:
                start_times.append(timestamps[0])

    if len(start_times) >= 2:
        spread_ms = (max(start_times) - min(start_times)) * 1000
        if spread_ms > 100:
            logger.warning(
                f"Sync quality warning for '{stage}' stage: "
                f"start time spread across instances is {spread_ms:.1f}ms (>100ms)"
            )


def compute_instance_stats(durations: List[float], start_timestamps: List[float]) -> Dict:
    """Compute summary statistics for a single instance's measurement."""
    arr = np.array(durations)
    first_start = start_timestamps[0] if start_timestamps else 0.0
    last_end_ts = start_timestamps[-1] + durations[-1] / 1000 if durations else 0.0
    total_actual_time_sec = last_end_ts - first_start if start_timestamps else 0.0

    return {
        "mean_duration_ms": float(np.mean(arr)),
        "std_duration_ms": float(np.std(arr)),
        "median_duration_ms": float(np.median(arr)),
        "p01_duration_ms": float(np.percentile(arr, 1)),
        "p25_duration_ms": float(np.percentile(arr, 25)),
        "p75_duration_ms": float(np.percentile(arr, 75)),
        "p99_duration_ms": float(np.percentile(arr, 99)),
        "first_iteration_ms": float(arr[0]),
        "total_actual_time_sec": float(total_actual_time_sec),
        "first_start_ts": float(first_start),
    }


def aggregate_stage_results(
    instance_outputs: List[Dict],
    stage: str,
    measurement_duration: float,
    core_assignments: List[str],
) -> Dict:
    """Aggregate per-instance results into a single stage result entry."""
    instances = []
    all_iterations = []

    for output in instance_outputs:
        if output.get("filtered"):
            continue
        stage_data = output.get("stages", {}).get(stage)
        if stage_data is None:
            continue

        iters = stage_data["iterations_completed"]
        durations = stage_data["duration_ms"]
        start_timestamps = stage_data["start_ts"]
        throughput = iters / measurement_duration if measurement_duration > 0 else 0.0

        instance_entry = {
            "instance_id": output["instance_id"],
            "taskset": core_assignments[output["instance_id"]],
            "iterations_completed": iters,
            "throughput_iterations_per_sec": float(throughput),
        }
        instance_entry.update(compute_instance_stats(durations, start_timestamps))

        instances.append(instance_entry)
        all_iterations.append(iters)

    if not instances:
        return {}

    throughputs = [inst["throughput_iterations_per_sec"] for inst in instances]

    aggregate = {
        "total_iterations": int(sum(all_iterations)),
        "total_throughput_iterations_per_sec": float(sum(throughputs)),
        "mean_throughput_per_instance": float(np.mean(throughputs)),
        "std_throughput_per_instance": float(np.std(throughputs)),
        "min_iterations_per_instance": int(min(all_iterations)),
        "max_iterations_per_instance": int(max(all_iterations)),
        "measurement_wall_time_sec": measurement_duration,
    }

    return {"instances": instances, "aggregate": aggregate}


def run_single_throughput_case(
    bench_case: BenchCase,
    filters: List[BenchCase],
    num_instances: int,
    cores_per_instance: int,
    measurement_duration: float,
    emergency_timeout: float,
    log_level: str,
) -> Tuple[int, List[Dict]]:
    """Run a single benchmark case in throughput mode."""
    # Preload dataset in parent process to avoid cache race condition
    # when multiple workers try to download/generate and save simultaneously
    from ..datasets import load_data

    logger.info("Preloading dataset in parent process to populate cache")
    load_data(bench_case)

    numa_conf = get_numa_cpus_conf()
    core_assignments = compute_core_assignments(
        num_instances, cores_per_instance, numa_conf or None
    )

    logger.info(
        f"Core assignments for {num_instances} instances: {core_assignments}"
    )

    server, port = create_server()
    logger.debug(f"Barrier server listening on localhost:{port}")

    bench_case_str = json.dumps(bench_case).replace(" ", "")
    filters_str = json.dumps({"filters": filters}).replace(" ", "")

    processes = []
    for i in range(num_instances):
        cmd = (
            f"numactl --physcpubind={core_assignments[i]} --localalloc "
            f"python -m sklbench.benchmarks.throughput_worker "
            f"--bench-case {bench_case_str} "
            f"--filters {filters_str} "
            f"--log-level {log_level} "
            f"--instance-id {i} "
            f"--barrier-port {port} "
            f"--measurement-duration {measurement_duration}"
        )
        logger.debug(f"Launching instance {i}: {cmd}")
        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        processes.append(proc)

    try:
        connections = accept_and_wait(server, num_instances, b"ready", emergency_timeout)
        logger.info("All workers ready, starting measurement stages")

        stages = ["training", "inference"]
        stage_timeout = measurement_duration + 60

        for stage in stages:
            logger.info(f"Sending 'go' for {stage} stage")
            send_all(connections, b"go")
            wait_all(connections, b"done", stage_timeout)
            logger.info(f"All workers done with {stage} stage")

        for conn in connections:
            conn.close()

    except Exception as e:
        logger.error(f"Barrier synchronization failed: {e}")
        for proc in processes:
            proc.kill()
        server.close()
        return -1, []

    server.close()

    # Collect outputs
    instance_outputs = []
    return_code = 0
    for i, proc in enumerate(processes):
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            logger.warning(
                f"Instance {i} returned non-zero code={proc.returncode}.\n"
                f"stderr: {stderr}"
            )
            return_code = proc.returncode
            continue

        if stderr:
            logger.debug(f"Instance {i} stderr: {stderr}")

        try:
            output = json.loads(stdout)
            instance_outputs.append(output)
        except json.JSONDecodeError:
            logger.warning(f"Instance {i}: unable to parse stdout as JSON")
            return_code = -1

    if not instance_outputs:
        return return_code, []

    # Validate sync quality
    for stage in stages:
        validate_sync_quality(instance_outputs, stage)

    # Build result entries (one per stage)
    results = []
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator")
    library_name = get_bench_case_value(bench_case, "algorithm:library")

    from ..benchmarks.sklearn_estimator import estimator_to_task

    task = estimator_to_task(estimator_name)

    # Get quality metrics from first instance (all should be similar)
    quality_metrics = instance_outputs[0].get("quality_metrics", {})
    final_estimator_params = instance_outputs[0].get("estimator_params", {})

    # Dataset info from bench_case
    from ..utils.bench_case import get_data_name

    dataset_name = get_data_name(bench_case, shortened=True)

    for stage in stages:
        stage_result = aggregate_stage_results(
            instance_outputs, stage, measurement_duration, core_assignments
        )
        if not stage_result:
            continue

        # Find the method name from first instance
        method = "unknown"
        for output in instance_outputs:
            stage_data = output.get("stages", {}).get(stage)
            if stage_data:
                method = stage_data.get("method", "unknown")
                break

        # Flatten aggregate metrics to top-level for report compatibility
        aggregate = stage_result.pop("aggregate")
        instances_detail = stage_result.pop("instances")

        result_entry = {
            "mode": "throughput",
            "stage": stage,
            "method": method,
            "task": task,
            "estimator": estimator_name,
            "dataset": dataset_name,
            "library": library_name,
            "device": get_bench_case_value(bench_case, "algorithm:device"),
            "num_instances": num_instances,
            "cores_per_instance": cores_per_instance,
            "measurement_duration_seconds": measurement_duration,
        }
        result_entry.update(aggregate)
        result_entry.update(quality_metrics)
        result_entry["instances"] = instances_detail
        results.append(result_entry)

    return return_code, results


def run_throughput_benchmarks(
    bench_cases: List[BenchCase],
    filters: List[BenchCase],
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Union[Dict, List]]]:
    """Main entry point for throughput mode."""
    env_info = get_environment_info()
    environment_name = args.environment_name or hash_from_json_repr(env_info)

    results = []
    return_code = 0

    bench_cases_with_pbar = tqdm(bench_cases)
    for bench_case in bench_cases_with_pbar:
        bench_cases_with_pbar.set_description(
            custom_format(
                get_bench_case_name(bench_case, shortened=True), bcolor="HEADER"
            )
        )

        # All throughput parameters come from bench_case config
        num_instances = get_bench_case_value(bench_case, "bench:num_instances")
        cores_per_instance = get_bench_case_value(bench_case, "bench:cores_per_instance")
        measurement_duration = get_bench_case_value(
            bench_case, "bench:measurement_duration", 60.0
        )
        emergency_timeout = get_bench_case_value(
            bench_case, "bench:emergency_timeout", 3600.0
        )

        try:
            validate_throughput_args(
                num_instances, cores_per_instance, measurement_duration
            )
        except ValueError as e:
            logger.error(f"Invalid throughput parameters: {e}")
            return_code = -1
            if args.exit_on_error:
                break
            continue

        try:
            case_return_code, case_results = run_single_throughput_case(
                bench_case,
                filters,
                num_instances,
                cores_per_instance,
                measurement_duration,
                emergency_timeout,
                args.bench_log_level,
            )
            if case_return_code != 0:
                return_code = case_return_code
                if args.exit_on_error:
                    break
            for entry in case_results:
                entry["environment_name"] = environment_name
                results.append(entry)
        except KeyboardInterrupt:
            return_code = -1
            break
        except Exception as e:
            logger.error(f"Throughput case failed: {e}")
            return_code = -1
            if args.exit_on_error:
                break

    full_result = {
        "bench_cases": results,
        "environment": {environment_name: env_info},
    }
    return return_code, full_result
