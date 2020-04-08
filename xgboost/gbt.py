# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output,
    accuracy_score, rmse_score
)
import numpy as np
import xgboost as xgb
import os


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_xgb_predictions(y_pred, objective):
    if objective == 'multi:softprob':
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == 'binary:logistic':
        y_pred = y_pred.astype(np.int32)
    return y_pred


parser = argparse.ArgumentParser(description='xgboost gradient boosted trees '
                                             'benchmark')

parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of gradient boosted trees')
parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                         'to prevents overfitting')
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')
parser.add_argument('--min-child-weight', type=float, default=1,
                    help='Minimum sum of instance weight needed in a child')
parser.add_argument('--max-delta-step', type=float, default=0,
                    help='Maximum delta step we allow each leaf output to be')
parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')
parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')
parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')
parser.add_argument('--tree-method', type=str, required=True,
                    help='The tree construction algorithm used in XGBoost')
parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')
parser.add_argument('--grow-policy', type=str, default='depthwise',
                    help='Controls a way new nodes are added to the tree')
parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')
parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
parser.add_argument('--objective', type=str, required=True,
                    choices=('reg:squarederror', 'binary:logistic',
                             'multi:softmax', 'multi:softprob'),
                    help='Control a balance of positive and negative weights')
parser.add_argument('--count-dmatrix', default=False, action='store_true',
                    help='Count DMatrix creation in time measurements')

params = parse_args(parser)

# Load and convert data
X_train, X_test, y_train, y_test = load_data(params)

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
    'seed': params.seed
}

if params.threads != -1:
    xgb_params.update({'nthread': params.threads})

if 'OMP_NUM_THREADS' in os.environ.keys():
    xgb_params['nthread'] = int(os.environ['OMP_NUM_THREADS'])

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees')

if params.objective.startswith('reg'):
    task = 'regression'
    metric_name, metric_func = 'rmse', rmse_score
    columns += ('rmse', 'time')
else:
    task = 'classification'
    metric_name = 'accuracy[%]'
    metric_func = lambda y1, y2: 100 * accuracy_score(y1, y2)
    columns += ('n_classes', 'accuracy', 'time')
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))
    if params.n_classes > 2:
        xgb_params['num_class'] = params.n_classes

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
if params.count_dmatrix:
    def fit():
        dtrain = xgb.DMatrix(X_train, y_train)
        return xgb.train(xgb_params, dtrain, params.n_estimators)

    def predict():
        dtest = xgb.DMatrix(X_test, y_test)
        return booster.predict(dtest)
else:
    def fit():
        return xgb.train(xgb_params, dtrain, params.n_estimators)

    def predict():
        return booster.predict(dtest)

fit_time, booster = measure_function_time(fit, params=params)
y_pred = convert_xgb_predictions(booster.predict(dtrain), params.objective)
train_metric = metric_func(y_pred, y_train)

predict_time, y_pred = measure_function_time(predict, params=params)
test_metric = metric_func(
    convert_xgb_predictions(y_pred, params.objective), y_test)

print_output(library='xgboost', algorithm=f'gradient_boosted_trees_{task}',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['gbt.fit', 'gbt.predict'],
             times=[fit_time, predict_time], accuracy_type=metric_name,
             accuracies=[train_metric, test_metric], data=[X_train, X_test],
             alg_instance=booster)
