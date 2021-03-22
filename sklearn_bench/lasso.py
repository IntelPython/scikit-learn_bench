# ===============================================================================
# Copyright 2020-2021 Intel Corporation
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
# ===============================================================================

import argparse

import bench
from sklearn.linear_model import Lasso


parser = argparse.ArgumentParser(description='scikit-learn lasso regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=False,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--alpha', dest='alpha', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--maxiter', type=int, default=1000,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('--tol', type=float, default=0.0,
                    help='Tolerance for solver.')
params = bench.parse_args(parser)


# Load data
X_train, X_test, y_train, y_test = bench.load_data(params)

# Create our regression object
regr = Lasso(fit_intercept=params.fit_intercept, alpha=params.alpha,
             tol=params.tol, max_iter=params.maxiter, copy_X=False)

# Time fit
fit_time, _ = bench.measure_function_time(regr.fit, X_train, y_train, params=params)

# Time predict
predict_time, pred_train = bench.measure_function_time(
    regr.predict, X_train, params=params)

train_rmse = bench.rmse_score(pred_train, y_train)
pred_test = regr.predict(X_test)
test_rmse = bench.rmse_score(pred_test, y_test)

bench.print_output(library='sklearn', algorithm='lasso',
                   stages=['training', 'prediction'], params=params,
                   functions=['Lasso.fit', 'Lasso.predict'],
                   times=[fit_time, predict_time], accuracy_type='rmse',
                   accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
                   alg_instance=regr)
