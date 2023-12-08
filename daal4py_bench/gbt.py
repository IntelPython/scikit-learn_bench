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

import argparse
import logging
import sys

import bench
import numpy as np
from sklearn.utils import check_random_state
from sklearn import preprocessing
import daal4py as d4p


parser = argparse.ArgumentParser(
    description='daal4py gradient boosted trees benchmark')

parser.add_argument('--max_bins', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
parser.add_argument('--min_bin_size', type=int, default=5,
                    help='Minimum size of discrete bins')
parser.add_argument('--max_tree_depth', type=int, default=8,
                    help='Maximum depth of a tree')
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--n_estimators', type=int, default=100,
                    help='The number of gradient boosted trees')
parser.add_argument('--reg_lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--split_method', type=str, required=False,
                    default='inexact',
                    help='The split algorithm used in daal4py')
parser.add_argument('--shrinkage', type=float, default=0.3,
                    help='Shrinkage rate')
parser.add_argument('--min_split_loss', type=float, default=0,
                    help='Minimal spilit loss')
parser.add_argument('--observations_per_tree_fraction', type=int, default=1,
                    help='Observations per tree fraction')
parser.add_argument('--features_per_node', type=int, default=0,
                    help='Features per node')
parser.add_argument('--min_observations_in_leaf_node', type=int, default=5,
                    help='Min observations in leaf node')
parser.add_argument('--memory_saving_mode', type=bool, default=False,
                    help='Enable memory saving mode')
parser.add_argument('--random_state', type=str, default=None,
                    help='Pass random state')
parser.add_argument('--objective', type=str, default="reg:squarederror",
                    help='Objective function')
parser.add_argument('--fptype', type=str, default="float",
                    help='FPType to use')

params = bench.parse_args(parser)

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)

if np.isnan(X_test.values).any():
    logging.warning('Nan values aren not supported in GBT DAAL fit yet')
    sys.exit(1)

# Get random seed
rs_ = check_random_state(params.random_state)
seed_ = rs_.randint(0, 2**31)

d4p_params = {
    'split_method': params.split_method,
    'n_estimators': params.n_estimators,
    'max_tree_depth': params.max_tree_depth,
    'shrinkage': params.shrinkage,
    'min_split_loss': params.min_split_loss,
    'reg_lambda': params.reg_lambda,
    'objective': params.objective,
    'fptype': params.fptype,
    'observations_per_tree_fraction': params.observations_per_tree_fraction,
    'features_per_node': params.features_per_node,
    'min_observations_in_leaf_node': params.min_observations_in_leaf_node,
    'memory_saving_mode': params.memory_saving_mode,
    'max_bins': params.max_bins,
    'min_bin_size': params.min_bin_size,
    'random_state': params.random_state}

if d4p_params["objective"].startswith('reg'):
    task = "regression"
    metric_name, metric_func = 'rmse', bench.rmse_score
    train_algo = d4p.gbt_regression_training(
      fptype=d4p_params["fptype"],
      splitMethod=d4p_params["split_method"],
      maxIterations=d4p_params["n_estimators"],
      maxTreeDepth=d4p_params["max_tree_depth"],
      shrinkage=d4p_params["shrinkage"],
      minSplitLoss=d4p_params["min_split_loss"],
      lambda_=d4p_params["reg_lambda"],
      observationsPerTreeFraction=d4p_params["observations_per_tree_fraction"],
      featuresPerNode=d4p_params["features_per_node"],
      minObservationsInLeafNode=d4p_params["min_observations_in_leaf_node"],
      memorySavingMode=d4p_params["memory_saving_mode"],
      maxBins=d4p_params["max_bins"],
      minBinSize=d4p_params["min_bin_size"],
      engine=d4p.engines_mcg59(seed=seed_))
else:
    task = "classification"
    metric_name = 'accuracy'
    metric_func = bench.accuracy_score
    le = preprocessing.LabelEncoder()
    le.fit(y_train)

    n_classes = len(le.classes_)
    # Covtype has one class more than there is in train
    if params.dataset_name == 'covtype':
        n_classes += 1
    n_iterations = d4p_params["n_estimators"]
    train_algo = d4p.gbt_classification_training(
      fptype=d4p_params["fptype"],
      nClasses=n_classes,
      splitMethod=d4p_params["split_method"],
      maxIterations=n_iterations,
      maxTreeDepth=d4p_params["max_tree_depth"],
      shrinkage=d4p_params["shrinkage"],
      minSplitLoss=d4p_params["min_split_loss"],
      lambda_=d4p_params["reg_lambda"],
      observationsPerTreeFraction=d4p_params["observations_per_tree_fraction"],
      featuresPerNode=d4p_params["features_per_node"],
      minObservationsInLeafNode=d4p_params["min_observations_in_leaf_node"],
      memorySavingMode=d4p_params["memory_saving_mode"],
      maxBins=d4p_params["max_bins"],
      minBinSize=d4p_params["min_bin_size"],
      engine=d4p.engines_mcg59(seed=seed_))


def fit(X_train, y_train):
    return train_algo.compute(X_train, y_train).model


def predict(X_test):  # type: ignore
    if task == "regression":
        predict_algo = d4p.gbt_regression_prediction(
            fptype=d4p_params["fptype"])
    else:
        predict_algo = d4p.gbt_classification_prediction(
            fptype=d4p_params["fptype"],
            nClasses=n_classes,
            resultsToEvaluate="computeClassLabels")
    return predict_algo.compute(X_test, booster).prediction.ravel()


fit_time, booster = bench.measure_function_time(
    fit, X_train, y_train, params=params)
train_metric = metric_func(
        predict(X_train), y_train)

predict_time, y_pred = bench.measure_function_time(
    predict, X_test, params=params)
test_metric = metric_func(y_pred, y_test)

bench.print_output(
                library='daal4py',
                algorithm=f'gradient_boosted_trees_{task}',
                stages=['training', 'prediction'],
                params=params, functions=['gbt.fit', 'gbt.predict'],
                times=[fit_time, predict_time], metric_type=metric_name,
                metrics=[train_metric, test_metric],
                data=[X_train, X_test],
                alg_params=d4p_params)
