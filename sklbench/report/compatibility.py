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

import math

import numpy as np
import pandas as pd

from ..utils.logger import logger


def transform_results_to_compatible(results: pd.DataFrame):
    # sklearn and sklearnex compatibility
    if (results["library"] == "sklearnex").any():
        # delete extra columns related to sklearnex only
        results.drop(
            inplace=True,
            errors="ignore",
            columns=[
                "max_bins",
                "min_bin_size",
            ],
        )
    # cuML compatibility
    if (
        (results["library"] == "cuml")
        | (results["library"] == "raft")
        | (results["library"] == "faiss")
    ).any():
        logger.info(
            "Found cuML, RAFT or FAISS entries in provided results. They will be "
            "filtered and transformed to make all entries compatible "
            "assuming config entries are aligned between cuML and other frameworks."
        )
        # delete extra columns related to cuML only or sklearn only
        results.drop(
            inplace=True,
            errors="ignore",
            columns=[
                # sklearn common
                "n_jobs",
                # cuML common
                "output_type",
                # cuML OR sklearn (dependent on algorithm)
                "random_state",
                "verbose",
                "normalize",
                "copy_x",
                "copy_X",
                "warm_start",
                # sklearn DBSCAN
                "leaf_size",
                # cuML DBSCAN
                "max_mbytes_per_batch",
                "calc_core_sample_indices",
                # cuML KMeans
                "oversampling_factor",
                "max_samples_per_batch",
                # sklearn kNN
                "leaf_size",
                "radius",
                # sklearn LinearRegression
                "positive",
                "precompute",
                # sklearn LogisticRegression
                "dual",
                "intercept_scaling",
                "multi_class",
                # cuML LogisticRegression
                "linesearch_max_iter",
                # sklearn PCA
                "n_oversamples",
                "power_iteration_normalizer",
                # cuML TSNE
                "late_exaggeration",
                "learning_rate_method",
                "perplexity_max_iter",
                "exaggeration_iter",
                "pre_momentum",
                "post_momentum",
                "square_distances",
                # sklearn SVM
                "break_ties",
                "shrinking",
                # cuML SVM
                "nochange_steps",
                # sklearn[ex] Ensemble
                "ccp_alpha",
                "max_bins",
                "min_bin_size",
                "min_weight_fraction_leaf",
                "oob_score",
                # cuml Ensemble
                "n_bins",
                "accuracy_metric",
                "max_batch_size",
                "n_streams",
                # NearestNeighbors emulators
                "n_lists",
                "n_probes",
                "m_subvectors",
                "n_bits",
                "intermediate_graph_degree",
                "graph_degree",
            ],
        )
        # DBSCAN parameters renaming
        cuml_dbscan_index = (results["estimator"] == "DBSCAN") & (
            results["library"] == "cuml"
        )
        if cuml_dbscan_index.any():
            results.loc[cuml_dbscan_index, "algorithm"] = "brute"
        # KMeans parameters renaming
        cuml_kmeans_index = (results["estimator"] == "KMeans") & (
            results["library"] == "cuml"
        )
        if cuml_kmeans_index.any():
            results.loc[cuml_kmeans_index, "algorithm"] = "lloyd"
            results.loc[
                cuml_kmeans_index & (results["init"] == "scalable-k-means++"), "init"
            ] = "k-means++"
        # Linear models parameters renaming
        linear_index = (
            (results["estimator"] == "LinearRegression")
            | (results["estimator"] == "Ridge")
            | (results["estimator"] == "Lasso")
            | (results["estimator"] == "ElasticNet")
        ) & (
            (results["library"] == "cuml")
            | (results["library"] == "sklearn")
            | (results["library"] == "sklearnex")
        )
        if linear_index.any():
            results.loc[linear_index, "algorithm"] = np.nan
            results.loc[linear_index, "solver"] = np.nan

        sklearn_ridge_index = (results["estimator"] == "Ridge") & (
            (results["library"] == "sklearn") | (results["library"] == "sklearnex")
        )
        if sklearn_ridge_index.any():
            results.loc[sklearn_ridge_index, "tol"] = np.nan

        cuml_logreg_index = (results["estimator"] == "LogisticRegression") & (
            results["library"] == "cuml"
        )
        if cuml_logreg_index.any():
            lbfgs_solver_index = (
                cuml_logreg_index
                & (results["solver"] == "qn")
                & ((results["penalty"] == "none") | (results["penalty"] == "l2"))
            )
            if lbfgs_solver_index.any():
                results.loc[lbfgs_solver_index, "solver"] = "lbfgs"
        # TSNE parameters renaming
        cuml_tsne_index = (results["estimator"] == "TSNE") & (
            results["library"] == "cuml"
        )
        if cuml_tsne_index.any():
            results.loc[cuml_tsne_index, "n_neighbors"] = np.nan
        # SVC parameters renaming
        cuml_svc_index = (results["estimator"] == "SVC") & (results["library"] == "cuml")
        if cuml_svc_index.any():
            results.loc[cuml_svc_index, "decision_function_shape"] = results.loc[
                cuml_svc_index, "multiclass_strategy"
            ]
            results.loc[cuml_svc_index, "multiclass_strategy"] = np.nan
        # Ensemble parameters renaming
        cuml_rf_index = (
            (results["estimator"] == "RandomForestClassifier")
            | (results["estimator"] == "RandomForestRegressor")
        ) & (results["library"] == "cuml")
        if cuml_rf_index.any():
            gini_index = cuml_rf_index & (results["split_criterion"] == 0)
            if gini_index.any():
                results.loc[gini_index, "criterion"] = "gini"
                results.loc[gini_index, "split_criterion"] = np.nan
            mse_index = cuml_rf_index & (results["split_criterion"] == 2)
            if mse_index.any():
                results.loc[mse_index, "criterion"] = "squared_error"
                results.loc[mse_index, "split_criterion"] = np.nan
            inf_leaves_index = cuml_rf_index & (results["max_leaves"] == -1)
            if inf_leaves_index.any():
                results.loc[inf_leaves_index, "max_leaf_nodes"] = None
                results.loc[inf_leaves_index, "max_leaves"] = np.nan

    return results
