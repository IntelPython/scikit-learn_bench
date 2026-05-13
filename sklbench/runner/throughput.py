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
import socket
import subprocess
import time
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

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
            "--num-instances is required and must be >= 1 in throughput mode"
        )
    if cores_per_instance is None or cores_per_instance < 1:
        raise ValueError(
            "--cores-per-instance is required and must be >= 1 in throughput mode"
        )
    if measurement_duration <= 0:
        raise ValueError("--measurement-duration must be > 0")


def create_barrier_server() -> Tuple[socket.socket, int]:
    """Create a TCP server socket on localhost with OS-assigned port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", 0))
    server.listen(128)
    port = server.getsockname()[1]
    return server, port


def wait_for_workers_ready(
    server: socket.socket, num_instances: int, timeout: float
) -> List[socket.socket]:
    """Accept connections from all workers and wait for 'ready' message."""
    server.settimeout(timeout)
    connections = []
    for _ in range(num_instances):
        conn, _ = server.accept()
        data = b""
        while b"ready" not in data:
            chunk = conn.recv(1024)
            if not chunk:
                raise ConnectionError("Worker disconnected before sending 'ready'")
            data += chunk
        connections.append(conn)
    return connections


def send_go_to_all(connections: List[socket.socket]):
    """Send 'go' signal to all workers."""
    for conn in connections:
        conn.sendall(b"go")


def wait_for_workers_done(connections: List[socket.socket], timeout: float):
    """Wait for 'done' message from all workers."""
    for conn in connections:
        conn.settimeout(timeout)
        data = b""
        while b"done" not in data:
            chunk = conn.recv(1024)
            if not chunk:
                raise ConnectionError("Worker disconnected before sending 'done'")
            data += chunk


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
    full_logs: bool = False,
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

        if full_logs:
            instance_entry["start_ts"] = start_timestamps
            instance_entry["duration_ms"] = durations

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
    full_logs: bool = False,
) -> Tuple[int, List[Dict]]:
    """Run a single benchmark case in throughput mode."""
    numa_conf = get_numa_cpus_conf()
    core_assignments = compute_core_assignments(
        num_instances, cores_per_instance, numa_conf if numa_conf else None
    )

    logger.info(
        f"Core assignments for {num_instances} instances: {core_assignments}"
    )

    server, port = create_barrier_server()
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
        # Wait for all workers to be ready (prep phase - unlimited, but bounded by emergency timeout)
        connections = wait_for_workers_ready(server, num_instances, emergency_timeout)
        logger.info("All workers ready, starting measurement stages")

        # Determine which stages exist
        estimator_methods_training = get_bench_case_value(
            bench_case, "algorithm:estimator_methods:training", None
        )
        estimator_methods_inference = get_bench_case_value(
            bench_case, "algorithm:estimator_methods:inference", None
        )
        stages = []
        if estimator_methods_training is not None:
            stages = ["training", "inference"]
        else:
            # default stages
            stages = ["training", "inference"]

        stage_timeout = measurement_duration + 60  # extra time for one stage

        for stage in stages:
            logger.info(f"Sending 'go' for {stage} stage")
            send_go_to_all(connections)
            wait_for_workers_done(connections, stage_timeout)
            logger.info(f"All workers done with {stage} stage")

        # Close barrier connections
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
    from .commands_helper import generate_benchmark_command

    from ..benchmarks.sklearn_estimator import estimator_to_task

    task = estimator_to_task(estimator_name)

    # Get quality metrics from first instance (all should be similar)
    quality_metrics = instance_outputs[0].get("quality_metrics", {})
    final_estimator_params = instance_outputs[0].get("estimator_params", {})

    for stage in stages:
        stage_result = aggregate_stage_results(
            instance_outputs, stage, measurement_duration, core_assignments, full_logs
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

        result_entry = {
            "mode": "throughput",
            "stage": stage,
            "method": method,
            "task": task,
            "estimator": estimator_name,
            "library": library_name,
            "device": get_bench_case_value(bench_case, "algorithm:device"),
            "num_instances": num_instances,
            "cores_per_instance": cores_per_instance,
            "measurement_duration_seconds": measurement_duration,
        }
        result_entry.update(quality_metrics)
        result_entry.update(final_estimator_params)
        result_entry.update(stage_result)
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

    # Resolve global defaults from CLI
    default_num_instances = args.num_instances
    default_cores_per_instance = args.cores_per_instance
    default_measurement_duration = args.measurement_duration
    default_emergency_timeout = args.emergency_timeout
    full_logs = args.throughput_full_logs

    results = []
    return_code = 0

    bench_cases_with_pbar = tqdm(bench_cases)
    for bench_case in bench_cases_with_pbar:
        bench_cases_with_pbar.set_description(
            custom_format(
                get_bench_case_name(bench_case, shortened=True), bcolor="HEADER"
            )
        )

        # Per-case config overrides CLI defaults
        num_instances = get_bench_case_value(
            bench_case, "bench:num_instances", default_num_instances
        )
        cores_per_instance = get_bench_case_value(
            bench_case, "bench:cores_per_instance", default_cores_per_instance
        )
        measurement_duration = get_bench_case_value(
            bench_case, "bench:measurement_duration", default_measurement_duration
        )
        emergency_timeout = get_bench_case_value(
            bench_case, "bench:emergency_timeout", default_emergency_timeout
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
                full_logs,
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
