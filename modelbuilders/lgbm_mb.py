# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import daal4py
import numpy as np
from os import environ
from timeit import default_timer as timer
from typing import Tuple
import lightgbm as lgbm
from bench import get_accuracy, load_data, measure_function_time, parse_args, print_output, read_csv, rmse_score


parser = argparse.ArgumentParser(description='lightgbm gbt + model transform + daal predict benchmark')

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

params = parse_args(parser)

X_train, X_test, y_train, y_test = load_data(params)

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

columns: Tuple[str, ...] = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size', 'num_trees')

if params.objective.startswith('reg'):
    task = 'regression'
    metric_name, metric_func = 'rmse', rmse_score
    columns += ('rmse', 'time')
else:
    task = 'classification'
    metric_name, metric_func = 'accuracy[%]', get_accuracy
    columns += ('n_classes', 'accuracy', 'time')
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))
    if params.n_classes > 2:
        lgbm_params['num_class'] = params.n_classes

t_creat_train, lgbm_train = measure_function_time(lgbm.Dataset, X_train, y_train, params=params, 
                                                    free_raw_data=False)

t_creat_test, lgbm_test = measure_function_time(lgbm.Dataset, X_test, y_test, params=params, 
                                                reference=lgbm_train, free_raw_data=False)

t_train, model_lgbm = measure_function_time(lgbm.train, lgbm_params,  lgbm_train, params=params,
                        num_boost_round=params.n_estimators, valid_sets=lgbm_train,
                        verbose_eval=False)
y_train_pred = model_lgbm.predict(X_train)
train_metric = metric_func(y_train, y_train_pred)

t_lgbm_pred, y_test_pred = measure_function_time(model_lgbm.predict, X_test, params=params)
test_metric_xgb = metric_func(y_test, y_test_pred)

t_trans, model_daal = measure_function_time(daal4py.get_gbt_model_from_lightgbm, model_lgbm, params=params)

if hasattr(params, 'n_classes'):
    predict_algo = daal4py.gbt_classification_prediction(nClasses=params.n_classes, 
        resultsToEvaluate='computeClassLabels', fptype='float')
    t_daal_pred, daal_pred = measure_function_time(predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)
else:
    predict_algo = daal4py.gbt_regression_prediction()
    t_daal_pred, daal_pred = measure_function_time(predict_algo.compute, X_test, model_daal, params=params)
    test_metric_daal = metric_func(y_test, daal_pred.prediction)

print_output(library='modelbuilders', algorithm=f'lightgbm_{task}_and_modelbuilder',
             stages=['lgbm_train_matrix_create', 'lgbm_test_matrix_create', 'lgbm_training',
                'lgbm_prediction', 'lgbm_to_daal_conv', 'daal_prediction'],
             columns=columns, params=params, functions=['lgbm_dataset', 'lgbm_dataset', 'lgbm_train',
                'lgbm_predict', 'lgbm_to_daal', 'daal_compute'],
             times=[t_creat_train, t_creat_test, t_train, t_lgbm_pred, t_trans, t_daal_pred],
             accuracy_type=metric_name, accuracies=[0, 0, train_metric, test_metric_xgb, 0, test_metric_daal],
             data=[X_train, X_test, X_train, X_test, X_train, X_test])