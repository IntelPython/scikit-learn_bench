# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import (
    parse_args, measure_function_time, load_data, print_output, rmse_score
)
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description='scikit-learn linear regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")

# Load data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'])

# Create our regression object
regr = LinearRegression(fit_intercept=params.fit_intercept,
                        n_jobs=params.n_jobs, copy_X=False)

# Time fit
fit_time, _ = measure_function_time(regr.fit, X_train, y_train, params=params)

# Time predict
predict_time, yp = measure_function_time(regr.predict, X_test, params=params)

test_rmse = rmse_score(yp, y_test)
yp = regr.predict(X_train)
train_rmse = rmse_score(yp, y_train)

print_output(library='sklearn', algorithm='linear_regression',
             stages=['training', 'prediction'],
             params=params, functions=['Linear.fit', 'Linear.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
             alg_instance=regr)
