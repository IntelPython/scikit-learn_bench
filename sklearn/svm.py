# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str, convert_data
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_optimal_cache_size(n_features, dtype=np.double, max_cache=64):
    '''
    Get an optimal cache size for sklearn.svm.SVC.

    Parameters
    ----------
    n_features : int
        Number of features in the dataset
    dtype : dtype-like, optional (default np.double)
        dtype to use for computing cache size
    max_cache : int, optional (default 64)
        Maximum cache size, in gigabytes
    '''

    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = byte_size * (n_features ** 2)
    one_gb = 2 ** 30
    max_cache_bytes = max_cache * one_gb
    if optimal_cache_size_bytes > max_cache_bytes:
        return max_cache_bytes
    else:
        return optimal_cache_size_bytes


parser = argparse.ArgumentParser(description='scikit-learn SVM benchmark')

parser.add_argument('-x', '--filex', '--fileX', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with features, in NPY format')
parser.add_argument('-y', '--filey', '--fileY', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with labels, in NPY format')
parser.add_argument('-C', dest='C', type=float, default=0.01,
                    help='SVM slack parameter')
parser.add_argument('--kernel', choices=('linear', 'rbf'),
                    default='linear', help='SVM kernel function')
parser.add_argument('--gamma', type=float, default=None,
                    help="Parameter for kernel='rbf'")
parser.add_argument('--maxiter', type=int, default=2000,
                    help='Maximum iterations for the iterative solver. '
                         '-1 means no limit.')
parser.add_argument('--max-cache-size', type=int, default=64,
                    help='Maximum cache size, in gigabytes, for SVM.')
parser.add_argument('--tol', type=float, default=1e-16,
                    help='Tolerance passed to sklearn.svm.SVC')
parser.add_argument('--no-shrinking', action='store_false', default=True,
                    dest='shrinking', help="Don't use shrinking heuristic")
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load data and cast to float64
X = np.load(params.filex.name).astype('f8')
y = np.load(params.filey.name).astype('f8')

X = convert_data(X, X.dtype, params.data_order, params.data_type)
y = convert_data(y, y.dtype, params.data_order, params.data_type)

if params.gamma is None:
    params.gamma = 'auto'

cache_size_bytes = get_optimal_cache_size(X.shape[0],
                                          max_cache=params.max_cache_size)
params.cache_size_mb = cache_size_bytes / 1024**2
params.n_classes = len(np.unique(y))

# Create our C-SVM classifier
clf = SVC(C=params.C, kernel=params.kernel, max_iter=params.maxiter,
          cache_size=params.cache_size_mb, tol=params.tol,
          shrinking=params.shrinking, gamma=params.gamma)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'kernel', 'cache_size_mb', 'C', 'sv_len', 'n_classes', 'accuracy',
           'time')
if params.data_type is "pandas":
    params.size = size_str(X.values.shape)
    params.dtype = X.values.dtype
else:
    params.size = size_str(X.shape)
    params.dtype = X.dtype

print_header(columns, params)

# Time fit and predict
fit_time, _ = time_mean_min(clf.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
params.sv_len = clf.support_.shape[0]
print_row(columns, params, function='SVM.fit', time=fit_time)

predict_time, y_pred = time_mean_min(clf.predict, X,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
acc = 100 * accuracy_score(y_pred, y)
print_row(columns, params, function='SVM.predict', time=predict_time,
          accuracy=acc)
