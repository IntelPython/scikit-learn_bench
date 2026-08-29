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
import inspect
import json
import socket
import time
from typing import Dict, List, Tuple

from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.barrier import recv_until
from ..utils.bench_case import get_bench_case_value
from ..utils.config import bench_case_filter
from ..utils.custom_types import BenchCase
from ..utils.logger import logger
from ..utils.special_params import assign_case_special_values_on_run
from .sklearn_estimator import (
    estimator_to_task,
    get_estimator,
    get_estimator_methods,
    get_subset_metrics_of_estimator,
    validate_estimator_params,
)



def run_measurement_loop(
    func, args: tuple, measurement_duration: float
) -> Dict[str, List]:
    """Run func repeatedly for measurement_duration seconds, recording each iteration."""
    start_timestamps = []
    durations_ms = []
    end_time = time.time() + measurement_duration
    while time.time() < end_time:
        t0 = time.time()
        func(*args)
        t1 = time.time()
        start_timestamps.append(t0)
        durations_ms.append((t1 - t0) * 1000)
    return {"start_ts": start_timestamps, "duration_ms": durations_ms}


def prepare_estimator(bench_case: BenchCase) -> Tuple:
    """Load data, create estimator, return everything needed for measurement."""
    library_name = get_bench_case_value(bench_case, "algorithm:library")
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator")

    estimator_class = get_estimator(library_name, estimator_name)
    task = estimator_to_task(estimator_name)

    data, data_description = load_data(bench_case)
    (x_train, x_test, y_train, y_test), data_description = split_and_transform_data(
        bench_case, data, data_description
    )

    assign_case_special_values_on_run(
        bench_case, (x_train, y_train, x_test, y_test), data_description
    )

    estimator_params = get_bench_case_value(
        bench_case, "algorithm:estimator_params", dict()
    )
    estimator_params = validate_estimator_params(estimator_class, estimator_params)
    estimator_methods = get_estimator_methods(bench_case)

    return (
        estimator_class,
        estimator_params,
        estimator_methods,
        task,
        x_train,
        x_test,
        y_train,
        y_test,
        data_description,
    )


def get_method_and_args(estimator_instance, method_name, stage, x_train, x_test, y_train, y_test):
    """Get bound method and appropriate data arguments."""
    method_instance = getattr(estimator_instance, method_name)
    if "y" in list(inspect.signature(method_instance).parameters):
        if stage == "training":
            data_args = (x_train, y_train)
        else:
            data_args = (x_test, y_test)
    else:
        if stage == "training":
            data_args = (x_train,)
        else:
            data_args = (x_test,)
    return method_instance, data_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-case", required=True, type=str)
    parser.add_argument("--filters", required=True, type=str)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--barrier-port", required=True, type=int)
    parser.add_argument("--measurement-duration", required=True, type=float)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    bench_case = json.loads(args.bench_case)
    filters = json.loads(args.filters)["filters"]

    if not bench_case_filter(bench_case, filters):
        logger.warning("Benchmarking case was filtered.")
        print(json.dumps({"instance_id": args.instance_id, "filtered": True}))
        return

    # --- Preparation phase (unlimited time) ---
    (
        estimator_class,
        estimator_params,
        estimator_methods,
        task,
        x_train,
        x_test,
        y_train,
        y_test,
        data_description,
    ) = prepare_estimator(bench_case)

    estimator_instance = estimator_class(**estimator_params)

    # Warmup: run one fit to trigger JIT/allocations
    training_methods = estimator_methods.get("training", ["fit"])
    for method_name in training_methods:
        if hasattr(estimator_instance, method_name):
            method_instance, data_args = get_method_and_args(
                estimator_instance, method_name, "training",
                x_train, x_test, y_train, y_test
            )
            method_instance(*data_args)
            break

    # --- Connect to barrier ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", args.barrier_port))
    sock.sendall(b"ready")

    # --- Measurement stages ---
    stages_results = {}

    for stage in ["training", "inference"]:
        methods = estimator_methods.get(stage, [])
        available_methods = [m for m in methods if hasattr(estimator_instance, m)]
        if not available_methods:
            continue

        # Wait for "go" signal from parent before each stage
        recv_until(sock, b"go")

        method_name = available_methods[0]
        method_instance, data_args = get_method_and_args(
            estimator_instance, method_name, stage,
            x_train, x_test, y_train, y_test
        )

        timing_data = run_measurement_loop(
            method_instance, data_args, args.measurement_duration
        )

        stages_results[stage] = {
            "method": method_name,
            "iterations_completed": len(timing_data["start_ts"]),
            "start_ts": timing_data["start_ts"],
            "duration_ms": timing_data["duration_ms"],
        }

        # Signal done to parent
        sock.sendall(b"done")

    # --- Compute quality metrics from final fitted model ---
    quality_metrics = {}
    quality_metrics.update(
        get_subset_metrics_of_estimator(
            task, "training", estimator_instance, (x_train, y_train)
        )
    )
    quality_metrics.update(
        get_subset_metrics_of_estimator(
            task, "inference", estimator_instance, (x_test, y_test)
        )
    )

    # Get final estimator params
    final_params = {}
    if hasattr(estimator_instance, "get_params"):
        final_params = estimator_instance.get_params()
        if "handle" in final_params:
            del final_params["handle"]

    sock.close()

    # --- Output ---
    output = {
        "instance_id": args.instance_id,
        "stages": stages_results,
        "quality_metrics": quality_metrics,
        "estimator_params": final_params,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
