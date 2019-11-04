# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row
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
                    help="Training method used by DAAL. 'normEqDense' selects"
                         "the normal equations method, while 'qrDense' selects"
                         "the method based on QR decomposition.")
params = parse_args(parser, size=(1000000, 50), dtypes=('f8', 'f4'),
                    loop_types=('fit', 'predict'), prefix='daal4py')

# Generate random data
X = np.random.rand(*params.shape).astype(params.dtype)
Xp = np.random.rand(*params.shape).astype(params.dtype)
y = np.random.rand(*params.shape).astype(params.dtype)


# Create our regression objects
def test_fit(X, y):
    regr_train = linear_regression_training(fptype=getFPType(X),
                                            method=params.method,
                                            interceptFlag=params.fit_intercept)
    return regr_train.compute(X, y)


def test_predict(Xp, model):
    regr_predict = linear_regression_prediction(fptype=getFPType(X))
    return regr_predict.compute(Xp, model)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'method', 'time')
print_header(columns, params)

# Time fit
fit_time, res = time_mean_min(test_fit, X, y,
                              outer_loops=params.fit_outer_loops,
                              inner_loops=params.fit_inner_loops,
                              time_limit=params.fit_time_limit,
                              verbose=params.verbose)
print_row(columns, params, function='Linear.fit', time=fit_time)

# Time predict
predict_time, yp = time_mean_min(test_predict, Xp, res.model,
                                 outer_loops=params.predict_outer_loops,
                                 inner_loops=params.predict_inner_loops,
                                 time_limit=params.predict_time_limit,
                                 verbose=params.verbose)
print_row(columns, params, function='Linear.predict', time=predict_time)
