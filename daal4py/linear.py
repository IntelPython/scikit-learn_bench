# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict,
    rmse_score
)
from daal4py import linear_regression_training, linear_regression_prediction
from daal4py.sklearn.utils import getFPType
import numpy as np

parser = argparse.ArgumentParser(description='daal4py linear regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--method', default='normEqDense',
                    choices=('normEqDense', 'qrDense'),
                    help='Training method used by DAAL. "normEqDense" selects'
                         'the normal equations method, while "qrDense" selects'
                         'the method based on QR decomposition.')
params = parse_args(parser, size=(1000000, 50),
                    loop_types=('fit', 'predict'), prefix='daal4py')

# Generate random data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'], add_dtype=True,
    label_2d=True if params.file_X_train is not None else False)


# Create our regression objects
def test_fit(X, y):
    regr_train = linear_regression_training(fptype=getFPType(X),
                                            method=params.method,
                                            interceptFlag=params.fit_intercept)
    return regr_train.compute(X, y)


def test_predict(Xp, model):
    regr_predict = linear_regression_prediction(fptype=getFPType(Xp))
    return regr_predict.compute(Xp, model)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'method', 'time')

# Time fit
fit_time, res = time_mean_min(test_fit, X_train, y_train,
                              outer_loops=params.fit_outer_loops,
                              inner_loops=params.fit_inner_loops,
                              goal_outer_loops=params.fit_goal,
                              time_limit=params.fit_time_limit,
                              verbose=params.verbose)

# Time predict
predict_time, pres = time_mean_min(test_predict, X_test, res.model,
                                  outer_loops=params.predict_outer_loops,
                                  inner_loops=params.predict_inner_loops,
                                  goal_outer_loops=params.predict_goal,
                                  time_limit=params.predict_time_limit,
                                  verbose=params.verbose)

if params.output_format == 'csv':
    output_csv(columns, params, functions=['Linear.fit', 'Linear.predict'],
               times=[fit_time, predict_time])
elif params.output_format == 'json':
    import json

    test_rmse = rmse_score(pres.prediction, y_test)
    pres = test_predict(X_train, res.model)
    train_rmse = rmse_score(pres.prediction, y_train)

    result = gen_basic_dict('daal4py', 'linear_regression',
                            'training', params, X_train)
    result.update({
        'time[s]': fit_time,
        'rmse': train_rmse
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('daal4py', 'linear_regression',
                            'prediction', params, X_test)
    result.update({
        'time[s]': predict_time,
        'rmse': test_rmse
    })
    print(json.dumps(result, indent=4))
