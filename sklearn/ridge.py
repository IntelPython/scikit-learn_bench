# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, convert_data
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser(description='scikit-learn ridge regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--solver', default='auto',
                    help='Solver used for training')
params = parse_args(parser, size=(1000000, 50), dtypes=('f8', 'f4'),
                    loop_types=('fit', 'predict'))

# Generate random data
X = np.random.rand(*params.shape).astype(params.dtype)
Xp = np.random.rand(*params.shape).astype(params.dtype)
y = np.random.rand(*params.shape).astype(params.dtype)

X = convert_data(X, X.dtype, params.data_order, params.data_type)
Xp = convert_data(Xp, Xp.dtype, params.data_order, params.data_type)
y = convert_data(y, y.dtype, params.data_order, params.data_type)

# Create our regression object
regr = Ridge(fit_intercept=params.fit_intercept,
             solver=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

print_header(columns, params)

# Time fit
fit_time, _ = time_mean_min(regr.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
print_row(columns, params, function='Ridge.fit', time=fit_time)

# Time predict
predict_time, yp = time_mean_min(regr.predict, Xp,
                                 outer_loops=params.predict_outer_loops,
                                 inner_loops=params.predict_inner_loops,
                                 goal_outer_loops=params.predict_goal,
                                 time_limit=params.predict_time_limit,
                                 verbose=params.verbose)
print_row(columns, params, function='Ridge.predict', time=predict_time)
