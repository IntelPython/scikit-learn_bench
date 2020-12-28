#===============================================================================
# Copyright 2020 Intel Corporation
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
#===============================================================================

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
import utils

import daal4py
import lightgbm as lgbm
import numpy as np
from os import environ

parser = argparse.ArgumentParser(
    description='lightgbm gbt + model transform + daal predict benchmark')

parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')
parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                         'to prevents overfitting')
parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
parser.add_argument('--max-delta-step', type=float, default=0,
                    help='Maximum delta step we allow each leaf output to be')
parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')
parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')
parser.add_argument('--min-child-weight', type=float, default=1,
                    help='Minimum sum of instance weight needed in a child')
parser.add_argument('--min-split-gain', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of gradient boosted trees')
parser.add_argument('--objective', type=str, required=True,
                    choices=('regression', 'binary', 'multiclass'),
                    help='Control a balance of positive and negative weights')
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')
parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')
parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')

params = bench.parse_args(parser)

X_train, X_test, y_train, y_test = bench.load_data(params)

lgbm_params = {
    'verbosity': -1,
    'learning_rate': params.learning_rate,
    'min_split_gain': params.min_split_gain,
    'max_depth': params.max_depth,
    'min_child_weight': params.min_child_weight,
    'max_delta_step': params.max_delta_step,
    'subsample': params.subsample,
    'colsample_bytree': params.colsample_bytree,
    'colsample_bynode': 1,
    'reg_lambda': params.reg_lambda,
    'reg_alpha': params.reg_alpha,
    'scale_pos_weight': params.scale_pos_weight,
    'max_leaves': params.max_leaves,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'seed': params.seed
}

if params.threads != -1:
    lgbm_params.update({'nthread': params.threads})

if 'OMP_NUM_THREADS' in environ.keys():
    lgbm_params['nthread'] = int(environ['OMP_NUM_THREADS'])

if params.objective.startswith('reg'):
    task = 'regression'
    metric_name, metric_func = 'rmse', bench.rmse_score
else:
    task = 'classification'
    metric_name, metric_func = 'accuracy[%]', utils.get_accuracy
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))
    if params.n_classes > 2:
        lgbm_params['num_class'] = params.n_classes

t_creat_train, lgbm_train = bench.measure_function_time(lgbm.Dataset, X_train,
                                                        y_train, params=params,
                                                        free_raw_data=False)

t_creat_test, lgbm_test = bench.measure_function_time(lgbm.Dataset, X_test, y_test,
                                                      params=params, reference=lgbm_train,
                                                      free_raw_data=False)

t_train, model_lgbm = bench.measure_function_time(lgbm.train, lgbm_params, lgbm_train,
                                                  params=params,
                                                  num_boost_round=params.n_estimators,
                                                  valid_sets=lgbm_train,
                                                  verbose_eval=False)
train_metric = None
if not X_train.equals(X_test):
    y_train_pred = model_lgbm.predict(X_train)
    train_metric = metric_func(y_train, y_train_pred)

t_lgbm_pred, y_test_pred = bench.measure_function_time(model_lgbm.predict, X_test,
                                                       params=params)
test_metric_lgbm = metric_func(y_test, y_test_pred)

t_trans, model_daal = bench.measure_function_time(
    daal4py.get_gbt_model_from_lightgbm, model_lgbm, params=params)

if hasattr(params, 'n_classes'):
    predict_algo = daal4py.gbt_classification_prediction(
        nClasses=params.n_classes, resultsToEvaluate='computeClassLabels', fptype='float')
    t_daal_pred, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)
else:
    predict_algo = daal4py.gbt_regression_prediction()
    t_daal_pred, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)

utils.print_output(
    library='modelbuilders', algorithm=f'lightgbm_{task}_and_modelbuilder',
    stages=['lgbm_train', 'lgbm_predict', 'daal4py_predict'],
    params=params, functions=['lgbm_dataset', 'lgbm_dataset', 'lgbm_train',
                              'lgbm_predict', 'lgbm_to_daal', 'daal_compute'],
    times=[t_creat_train, t_train, t_creat_test, t_lgbm_pred, t_trans, t_daal_pred],
    accuracy_type=metric_name, accuracies=[train_metric, test_metric_lgbm,
                                           test_metric_daal],
    data=[X_train, X_test, X_test])
