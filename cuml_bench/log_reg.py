#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
from cuml import LogisticRegression

parser = argparse.ArgumentParser(description='cuML logistic '
                                             'regression benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                    action='store_false', default=True,
                    help="Don't fit intercept")
parser.add_argument('--solver', default='qn', choices=('qn', 'owl'),
                    help='Solver to use.')
parser.add_argument('--linesearch-max-iter', type=int, default=50,
                    help='Maximum iterations per solver outer iteration')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--tol', type=float, default=1e-10,
                    help='Tolerance for solver. Default is 1e-10.')
params = bench.parse_args(parser)

# Load generated data
X_train, X_test, y_train, y_test = bench.load_data(params)

params.n_classes = y_train[y_train.columns[0]].nunique()

# Create our classifier object
clf = LogisticRegression(penalty='l2', C=params.C,
                         linesearch_max_iter=params.linesearch_max_iter,
                         fit_intercept=params.fit_intercept, verbose=params.verbose,
                         tol=params.tol,
                         max_iter=params.maxiter, solver=params.solver)

# Time fit and predict
fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
y_pred = clf.predict(X_train)
train_acc = 100 * bench.accuracy_score(y_pred, y_train)

predict_time, y_pred = bench.measure_function_time(
    clf.predict, X_test, params=params)
test_acc = 100 * bench.accuracy_score(y_pred, y_test)

bench.print_output(library='cuml', algorithm='logistic_regression',
                   stages=['training', 'prediction'], params=params,
                   functions=['LogReg.fit', 'LogReg.predict'],
                   times=[fit_time, predict_time], accuracy_type='accuracy[%]',
                   accuracies=[train_acc, test_acc], data=[X_train, X_test],
                   alg_instance=clf)
