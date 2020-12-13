# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, rmse_score,
    getFPType
)
import numpy as np
from daal4py import ridge_regression_training, ridge_regression_prediction


parser = argparse.ArgumentParser(description='daal4py ridge regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Regularization strength')
params = parse_args(parser, size=(1000000, 50), prefix='daal4py')

# Generate random data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'], add_dtype=True,
    label_2d=True if params.file_X_train is not None else False)


# Create our regression objects
def test_fit(X, y):
    regr_train = ridge_regression_training(
        fptype=getFPType(X), ridgeParameters=np.array([[params.alpha]]),
        interceptFlag=params.fit_intercept)
    return regr_train.compute(X, y)


def test_predict(Xp, model):
    regr_predict = ridge_regression_prediction(fptype=getFPType(Xp))
    return regr_predict.compute(Xp, model)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

# Time fit
fit_time, res = measure_function_time(
    test_fit, X_train, y_train, params=params)

# Time predict
predict_time, yp = measure_function_time(
    test_predict, X_test, res.model, params=params)

test_rmse = rmse_score(yp.prediction, y_test)
pres = test_predict(X_train, res.model)
train_rmse = rmse_score(pres.prediction, y_train)

print_output(library='daal4py', algorithm='ridge_regression',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['Ridge.fit', 'Ridge.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test])
