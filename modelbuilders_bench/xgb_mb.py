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
import daal4py
import numpy as np
from os import environ
import xgboost as xgb
import utils

parser = argparse.ArgumentParser(
    description='xgboost gbt + model transform + daal predict benchmark')

parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')
parser.add_argument('--count-dmatrix', default=False, action='store_true',
                    help='Count DMatrix creation in time measurements')
parser.add_argument('--enable-experimental-json-serialization', default=True,
                    choices=('True', 'False'), help='Use JSON to store memory snapshots')
parser.add_argument('--grow-policy', type=str, default='depthwise',
                    help='Controls a way new nodes are added to the tree')
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
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of gradient boosted trees')
parser.add_argument('--objective', type=str, required=True,
                    choices=('reg:squarederror', 'binary:logistic',
                             'multi:softmax', 'multi:softprob'),
                    help='Control a balance of positive and negative weights')
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')
parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')
parser.add_argument('--single-precision-histogram', default=False, action='store_true',
                    help='Build histograms instead of double precision')
parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')
parser.add_argument('--tree-method', type=str, required=True,
                    help='The tree construction algorithm used in XGBoost')

params = bench.parse_args(parser)

X_train, X_test, y_train, y_test = bench.load_data(params)

xgb_params = {
    'booster': 'gbtree',
    'verbosity': 0,
    'learning_rate': params.learning_rate,
    'min_split_loss': params.min_split_loss,
    'max_depth': params.max_depth,
    'min_child_weight': params.min_child_weight,
    'max_delta_step': params.max_delta_step,
    'subsample': params.subsample,
    'sampling_method': 'uniform',
    'colsample_bytree': params.colsample_bytree,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_lambda': params.reg_lambda,
    'reg_alpha': params.reg_alpha,
    'tree_method': params.tree_method,
    'scale_pos_weight': params.scale_pos_weight,
    'grow_policy': params.grow_policy,
    'max_leaves': params.max_leaves,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'seed': params.seed,
    'single_precision_histogram': params.single_precision_histogram,
    'enable_experimental_json_serialization':
        params.enable_experimental_json_serialization
}

if params.threads != -1:
    xgb_params.update({'nthread': params.threads})

if 'OMP_NUM_THREADS' in environ.keys():
    xgb_params['nthread'] = int(environ['OMP_NUM_THREADS'])

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
        xgb_params['num_class'] = params.n_classes

t_creat_train, dtrain = bench.measure_function_time(xgb.DMatrix, X_train,
                                                    params=params, label=y_train)
t_creat_test, dtest = bench.measure_function_time(xgb.DMatrix, X_test, params=params)


def fit(dmatrix=None):
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_train, y_train)
    return xgb.train(xgb_params, dmatrix, params.n_estimators)


def predict():
    dmatrix = xgb.DMatrix(X_test, y_test)
    return model_xgb.predict(dmatrix)


t_train, model_xgb = bench.measure_function_time(
    fit, None if params.count_dmatrix else dtrain, params=params)
train_metric = None
if not X_train.equals(X_test):
    y_train_pred = model_xgb.predict(dtrain)
    train_metric = metric_func(y_train, y_train_pred)

t_xgb_pred, y_test_pred = bench.measure_function_time(predict, params=params)
test_metric_xgb = metric_func(y_test, y_test_pred)

t_trans, model_daal = bench.measure_function_time(
    daal4py.get_gbt_model_from_xgboost, model_xgb, params=params)

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

utils.print_output(library='modelbuilders',
                   algorithm=f'xgboost_{task}_and_modelbuilder',
                   stages=['xgboost_train', 'xgboost_predict', 'daal4py_predict'],
                   params=params, functions=['xgb_dmatrix', 'xgb_dmatrix',
                                             'xgb_train', 'xgb_predict',
                                             'xgb_to_daal', 'daal_compute'],
                   times=[t_creat_train, t_train, t_creat_test, t_xgb_pred,
                          t_trans, t_daal_pred],
                   accuracy_type=metric_name,
                   accuracies=[train_metric, test_metric_xgb, test_metric_daal],
                   data=[X_train, X_test, X_test])
