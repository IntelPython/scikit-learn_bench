# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict,
    rmse_score
)
from cuml import LinearRegression

parser = argparse.ArgumentParser(description='scikit-learn linear regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--solver', default='eig', choices=('eig', 'svd'),
                    help='Solver used for training')
params = parse_args(parser, prefix='cuml', size=(1000000, 50),
                    loop_types=('fit', 'predict'))

# Load data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'])

# Create our regression object
regr = LinearRegression(fit_intercept=params.fit_intercept,
                        algorithm=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

# Time fit
fit_time, _ = time_mean_min(regr.fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time predict
predict_time, y_pred = time_mean_min(regr.predict, X_test,
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

    test_rmse = rmse_score(y_pred, y_test)
    y_pred = regr.predict(X_train)
    train_rmse = rmse_score(y_pred, y_train)

    result = gen_basic_dict('cuml', 'linear_regression',
                            'training', params, X_train, regr)
    result.update({
        'time[s]': fit_time,
        'rmse': train_rmse
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('cuml', 'linear_regression',
                            'prediction', params, X_test, regr)
    result.update({
        'time[s]': predict_time,
        'rmse': test_rmse
    })
    print(json.dumps(result, indent=4))
