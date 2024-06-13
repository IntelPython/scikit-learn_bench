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

import inspect
import io
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ..datasets import load_data
from ..datasets.transformer import split_and_transform_data
from ..utils.bench_case import get_bench_case_value
from ..utils.common import convert_to_numpy, custom_format, get_module_members
from ..utils.config import bench_case_filter
from ..utils.custom_types import BenchCase, Numeric, NumpyNumeric
from ..utils.logger import logger
from ..utils.measurement import measure_case
from ..utils.special_params import assign_case_special_values_on_run
from .common import enrich_result, main_template


def get_estimator(library_name: str, estimator_name: str):
    classes_map, _ = get_module_members(library_name.split("."))
    if estimator_name not in classes_map:
        raise ValueError(
            f"Unable to find {estimator_name} estimator in {library_name} module."
        )
    if len(classes_map[estimator_name]) != 1:
        logger.debug(
            f'List of estimator with name "{estimator_name}": '
            f"{classes_map[estimator_name]}"
        )
        logger.warning(
            f"Found {len(classes_map[estimator_name])} classes for "
            f'"{estimator_name}" estimator name. '
            f"Using first {classes_map[estimator_name][0]}."
        )
    estimator = classes_map[estimator_name][0]
    if not issubclass(estimator, BaseEstimator):
        logger.info(f"{estimator} estimator is not derived from sklearn's BaseEstimator")
    return estimator


def get_estimator_methods(bench_case: BenchCase) -> Dict[str, List[str]]:
    # default estimator methods
    estimator_methods = {
        "training": ["fit"],
        "inference": ["predict", "predict_proba", "transform"],
    }
    for stage in estimator_methods.keys():
        methods = get_bench_case_value(
            bench_case, f"algorithm:estimator_methods:{stage}", None
        )
        if methods is not None:
            estimator_methods[stage] = methods.split("|")
    return estimator_methods


def estimator_to_task(estimator_name: str) -> str:
    """Maps estimator name to machine learning task based on listed estimator postfixes"""
    with open(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "estimator_task_map.json"
        )
    ) as map_file:
        estimator_to_task_map = json.load(map_file)
    for task, postfixes_list in estimator_to_task_map.items():
        if any(map(lambda postfix: estimator_name.endswith(postfix), postfixes_list)):
            return task
    return "unknown"


def get_number_of_classes(estimator_instance, y):
    classes = getattr(estimator_instance, "classes_", None)
    class_weight = getattr(estimator_instance, "_class_weight", None)
    if classes is not None and hasattr(classes, "__len__"):
        return len(classes)
    elif class_weight is not None and hasattr(class_weight, "__len__"):
        return len(class_weight)
    else:
        return len(np.unique(y))


def get_subset_metrics_of_estimator(
    task, stage, estimator_instance, data
) -> Dict[str, float]:
    # brute kNN with transfer between training and inference stages
    # is required for recall metric calculation of search task
    global _brute_knn

    metrics = dict()
    # Note: use `x` and `y` when calling estimator methods,
    # and `x_compat` and `y_compat` for compatibility with sklearn metrics
    x, y = data
    x_compat, y_compat = list(map(lambda i: convert_to_numpy(i), data))
    if stage == "training":
        if hasattr(estimator_instance, "n_iter_"):
            iterations = estimator_instance.n_iter_
            if isinstance(iterations, Union[Numeric, NumpyNumeric].__args__):
                metrics.update({"iterations": int(iterations)})
            elif (
                hasattr(iterations, "__len__")
                and len(iterations) == 1
                and isinstance(iterations[0], Union[Numeric, NumpyNumeric].__args__)
            ):
                metrics.update({"iterations": int(iterations[0])})
    if task == "classification":
        y_pred = convert_to_numpy(estimator_instance.predict(x))
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_compat, y_pred)),
                "balanced accuracy": float(balanced_accuracy_score(y_compat, y_pred)),
            }
        )
        if hasattr(estimator_instance, "predict_proba") and not (
            hasattr(estimator_instance, "probability")
            and getattr(estimator_instance, "probability") == False
        ):
            y_pred_proba = convert_to_numpy(estimator_instance.predict_proba(x))
            metrics.update(
                {
                    "ROC AUC": float(
                        roc_auc_score(
                            y_compat,
                            (
                                y_pred_proba
                                if y_pred_proba.shape[1] > 2
                                else y_pred_proba[:, 1]
                            ),
                            multi_class="ovr",
                        )
                    ),
                    "logloss": float(log_loss(y_compat, y_pred_proba)),
                }
            )
    elif task == "regression":
        y_pred = convert_to_numpy(estimator_instance.predict(x))
        metrics.update(
            {
                "RMSE": float(mean_squared_error(y_compat, y_pred) ** 0.5),
                "R2": float(r2_score(y_compat, y_pred)),
            }
        )
    elif task == "decomposition":
        if "PCA" in str(estimator_instance):
            if hasattr(estimator_instance, "score"):
                metrics.update(
                    {"average log-likelihood": float(estimator_instance.score(x))}
                )
            if stage == "training" and hasattr(
                estimator_instance, "explained_variance_ratio_"
            ):
                metrics.update(
                    {
                        "1st component variance ratio": float(
                            estimator_instance.explained_variance_ratio_[0]
                        )
                    }
                )
    elif task == "clustering":
        if hasattr(estimator_instance, "inertia_"):
            # compute inertia manually using distances to cluster centers
            # provided by KMeans.transform
            metrics.update(
                {
                    "inertia": float(
                        np.power(
                            convert_to_numpy(estimator_instance.transform(x)).min(axis=1),
                            2,
                        ).sum()
                    )
                }
            )
        if hasattr(estimator_instance, "predict"):
            y_pred = convert_to_numpy(estimator_instance.predict(x))
            metrics.update(
                {
                    "Davies-Bouldin score": float(davies_bouldin_score(x_compat, y_pred)),
                    "homogeneity": float(homogeneity_score(y_compat, y_pred)),
                    "completeness": float(completeness_score(y_compat, y_pred)),
                }
            )
        if "DBSCAN" in str(estimator_instance) and stage == "training":
            labels = convert_to_numpy(estimator_instance.labels_)
            clusters = len(np.unique(labels[labels != -1]))
            metrics.update({"clusters": clusters})
            if clusters > 1:
                metrics.update(
                    {
                        "Davies-Bouldin score": float(
                            davies_bouldin_score(x_compat, labels)
                        )
                    }
                )
            if len(np.unique(y)) < 128:
                metrics.update(
                    {
                        "homogeneity": (
                            float(homogeneity_score(y_compat, labels))
                            if clusters > 1
                            else 0
                        ),
                        "completeness": (
                            float(completeness_score(y_compat, labels))
                            if clusters > 1
                            else 0
                        ),
                    }
                )
    elif task == "manifold":
        if hasattr(estimator_instance, "kl_divergence_") and stage == "training":
            metrics.update(
                {"Kullback-Leibler divergence": float(estimator_instance.kl_divergence_)}
            )
    elif task == "search":
        if stage == "training":
            from sklearn.neighbors import NearestNeighbors

            _brute_knn = NearestNeighbors(algorithm="brute").fit(x_compat)
        else:
            recall_degree = 10
            ground_truth_neighbors = _brute_knn.kneighbors(
                x_compat, recall_degree, return_distance=False
            )
            predicted_neighbors = convert_to_numpy(
                estimator_instance.kneighbors(x, recall_degree, return_distance=False)
            )
            n_relevant = 0
            for i in range(ground_truth_neighbors.shape[0]):
                n_relevant += len(
                    np.intersect1d(ground_truth_neighbors[i], predicted_neighbors[i])
                )
            recall = (
                n_relevant
                / ground_truth_neighbors.shape[0]
                / ground_truth_neighbors.shape[1]
            )
            metrics.update({f"recall@{recall_degree}": recall})
    if (
        hasattr(estimator_instance, "support_vectors_")
        and estimator_instance.support_vectors_ is not None
    ):
        metrics.update({"support vectors": len(estimator_instance.support_vectors_)})
    return metrics


def get_context(bench_case: BenchCase):
    sklearn_context, sklearnex_context = [
        get_bench_case_value(bench_case, f"algorithm:{library}_context", None)
        for library in ["sklearn", "sklearnex"]
    ]
    if sklearnex_context is not None:
        from sklearnex import config_context

        if sklearn_context is not None:
            logger.info(
                f"Updating sklearnex context {sklearnex_context} "
                f"with sklearn context {sklearn_context}"
            )
            sklearnex_context.update(sklearn_context)
        return config_context, sklearnex_context
    elif sklearn_context is not None:
        from sklearn import config_context

        return config_context, sklearn_context
    else:
        from contextlib import nullcontext

        return nullcontext, dict()


def sklearnex_logger_is_available() -> bool:
    try:
        sklex_version = tuple(map(int, version("scikit-learn-intelex").split(".")))
        # scikit-learn-intelex packages is still signed with build date
        return sklex_version > (20230510, 0)
    except PackageNotFoundError:
        return False


def get_sklearnex_logging_stream() -> io.StringIO:
    sklex_logger = logging.getLogger("sklearnex")
    sklex_logger.setLevel(logging.INFO)
    for handler in sklex_logger.handlers.copy():
        sklex_logger.removeHandler(handler)
    stream = io.StringIO()
    channel = logging.StreamHandler(stream)
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    channel.setFormatter(formatter)
    sklex_logger.addHandler(channel)
    return stream


def verify_patching(stream: io.StringIO, function_name) -> bool:
    acceleration_lines = 0
    fallback_lines = 0
    logs = stream.getvalue().split("\n")[:-1]
    for line in logs:
        if function_name in line:
            if "running accelerated version on" in line:
                acceleration_lines += 1
            if "fallback to original Scikit-learn" in line:
                fallback_lines += 1
    return acceleration_lines > 0 and fallback_lines == 0


def create_online_function(method_instance, data_args, batch_size):
    n_batches = data_args[0].shape[0] // batch_size

    if "y" in list(inspect.signature(method_instance).parameters):

        def ndarray_function(x, y):
            for i in range(n_batches):
                method_instance(
                    x[i * batch_size : (i + 1) * batch_size],
                    y[i * batch_size : (i + 1) * batch_size],
                )

        def dataframe_function(x, y):
            for i in range(n_batches):
                method_instance(
                    x.iloc[i * batch_size : (i + 1) * batch_size],
                    y.iloc[i * batch_size : (i + 1) * batch_size],
                )

    else:

        def ndarray_function(x):
            for i in range(n_batches):
                method_instance(x[i * batch_size : (i + 1) * batch_size])

        def dataframe_function(x):
            for i in range(n_batches):
                method_instance(x.iloc[i * batch_size : (i + 1) * batch_size])

    if "ndarray" in str(type(data_args[0])):
        return ndarray_function
    elif "DataFrame" in str(type(data_args[0])):
        return dataframe_function
    else:
        return f"Unknown {type(data_args[0])} input type for online execution mode"


def measure_sklearn_estimator(
    bench_case,
    task,
    estimator_class,
    estimator_methods,
    estimator_params,
    x_train,
    x_test,
    y_train,
    y_test,
):
    enable_modelbuilders = get_bench_case_value(
        bench_case, "algorithm:enable_modelbuilders", False
    )
    ensure_sklearnex_patching = get_bench_case_value(
        bench_case, "bench:ensure_sklearnex_patching", True
    )
    ensure_sklearnex_patching = (
        ensure_sklearnex_patching
        and sklearnex_logger_is_available()
        and (
            estimator_class.__module__.startswith("daal4py")
            or estimator_class.__module__.startswith("sklearnex")
        )
    )
    sklearnex_logging_stream = get_sklearnex_logging_stream()

    metrics = dict()
    estimator_instance = estimator_class(**estimator_params)
    for stage in estimator_methods.keys():
        for method in estimator_methods[stage]:
            if hasattr(estimator_instance, method):
                method_instance = getattr(estimator_instance, method)
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
                batch_size = get_bench_case_value(
                    bench_case, f"algorithm:batch_size:{stage}"
                )
                if batch_size is not None:
                    method_instance = create_online_function(
                        method_instance, data_args, batch_size
                    )
                # daal4py model builders enabling branch
                if enable_modelbuilders and stage == "inference":
                    import daal4py

                    daal_model = daal4py.mb.convert_model(
                        estimator_instance.get_booster()
                    )
                    method_instance = getattr(daal_model, method)

                metrics[method] = dict()
                (
                    metrics[method]["time[ms]"],
                    metrics[method]["time std[ms]"],
                    _,
                ) = measure_case(bench_case, method_instance, *data_args)
                if batch_size is not None:
                    metrics[method]["throughput[samples/ms]"] = (
                        (data_args[0].shape[0] // batch_size) * batch_size
                    ) / metrics[method]["time[ms]"]
                if ensure_sklearnex_patching:
                    full_method_name = f"{estimator_class.__name__}.{method}"
                    sklearnex_logging_stream.seek(0)
                    method_is_patched = verify_patching(
                        sklearnex_logging_stream, full_method_name
                    )
                    if not method_is_patched:
                        logger.warning(
                            f"{full_method_name} was not patched by sklearnex."
                        )

    quality_metrics = {
        "training": get_subset_metrics_of_estimator(
            task, "training", estimator_instance, (x_train, y_train)
        ),
        "inference": get_subset_metrics_of_estimator(
            task, "inference", estimator_instance, (x_test, y_test)
        ),
    }
    for method in metrics.keys():
        for stage in estimator_methods.keys():
            if method in estimator_methods[stage]:
                metrics[method].update(quality_metrics[stage])

    return metrics, estimator_instance


def main(bench_case: BenchCase, filters: List[BenchCase]):
    # get estimator class and ML task
    library_name = get_bench_case_value(bench_case, "algorithm:library")
    estimator_name = get_bench_case_value(bench_case, "algorithm:estimator")

    estimator_class = get_estimator(library_name, estimator_name)
    task = estimator_to_task(estimator_name)

    # load and transform data
    data, data_description = load_data(bench_case)
    (x_train, x_test, y_train, y_test), data_description = split_and_transform_data(
        bench_case, data, data_description
    )

    # assign special values
    assign_case_special_values_on_run(
        bench_case, (x_train, y_train, x_test, y_test), data_description
    )

    # get estimator parameters
    estimator_params = get_bench_case_value(
        bench_case, "algorithm:estimator_params", dict()
    )

    # get estimator methods for measurement
    estimator_methods = get_estimator_methods(bench_case)

    # benchmark case filtering
    if not bench_case_filter(bench_case, filters):
        logger.warning("Benchmarking case was filtered.")
        return list()

    # run estimator methods
    context_class, context_params = get_context(bench_case)
    with context_class(**context_params):
        metrics, estimator_instance = measure_sklearn_estimator(
            bench_case,
            task,
            estimator_class,
            estimator_methods,
            estimator_params,
            x_train,
            x_test,
            y_train,
            y_test,
        )

    result_template = {
        "task": task,
        "estimator": estimator_name,
    }
    result_template = enrich_result(result_template, bench_case)
    if "assume_finite" in context_params:
        result_template["assume_finite"] = context_params["assume_finite"]
    if hasattr(estimator_instance, "get_params"):
        estimator_params = estimator_instance.get_params()
    # note: "handle" is not JSON-serializable
    if "handle" in estimator_params:
        del estimator_params["handle"]
    logger.debug(f"Estimator parameters:\n{custom_format(estimator_params)}")
    result_template.update(estimator_params)

    data_descs = {
        "training": data_description["x_train"],
        "inference": data_description["x_test"],
    }
    for stage in estimator_methods.keys():
        data_descs[stage].update(
            {
                "batch_size": get_bench_case_value(
                    bench_case, f"algorithm:batch_size:{stage}"
                )
            }
        )
        if "n_classes" in data_description:
            data_descs[stage].update({"n_classes": data_description["n_classes"]})

    results = list()
    for method in metrics.keys():
        result = result_template.copy()
        for stage in estimator_methods.keys():
            if method in estimator_methods[stage]:
                result.update({"stage": stage, "method": method})
                result.update(data_descs[stage])
                result.update(metrics[method])
        results.append(result)

    return results


if __name__ == "__main__":
    main_template(main)
