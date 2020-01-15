# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str, convert_data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'regression benchmark')
parser.add_argument('-x', '--filex', '--fileX', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with features, in NPY format')
parser.add_argument('-y', '--filey', '--fileY', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with targets, in NPY format')

parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')

parser.add_argument('--use-sklearn-class', action='store_true',
                    help='Force use of '
                         'sklearn.ensemble.RandomForestRegressor')
parser.add_argument('--seed', type=int, default=12345,
                    help='Seed to pass as random_state to the class')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Get some RandomForestRegressor
if params.use_sklearn_class:
    from sklearn.ensemble import RandomForestRegressor
else:
    try:
        from daal4py.sklearn.ensemble import RandomForestRegressor
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

# Load data
X = np.load(params.filex.name)
y = np.load(params.filey.name)

X = convert_data(X, X.dtype, params.data_order, params.data_type)
y = convert_data(y, y.dtype, params.data_order, params.data_type)

# Create our random forest regressor
regr = RandomForestRegressor(n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             random_state=params.seed)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'time')
if params.data_type is "pandas":
    params.size = size_str(X.values.shape)
    params.dtype = X.values.dtype
else:
    params.size = size_str(X.shape)
    params.dtype = X.dtype

print_header(columns, params)

# Time fit and predict
fit_time, _ = time_mean_min(regr.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
print_row(columns, params, function='df_regr.fit', time=fit_time)

predict_time, y_pred = time_mean_min(regr.predict, X,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
print_row(columns, params, function='df_regr.predict', time=predict_time)
