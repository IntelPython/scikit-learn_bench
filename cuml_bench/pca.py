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

from cuml import PCA

parser = argparse.ArgumentParser(description='cuML PCA benchmark')
parser.add_argument('--svd-solver', type=str, default='full',
                    choices=['auto', 'full', 'jacobi'],
                    help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
params = bench.parse_args(parser)

# Load random data
X_train, X_test, _, _ = bench.load_data(params, generated_data=['X_train'])

if params.n_components is None:
    p, n = X_train.shape
    params.n_components = min((n, (2 + min((n, p))) // 3))

# Create our PCA object
pca = PCA(svd_solver=params.svd_solver, whiten=params.whiten,
          n_components=params.n_components)

# Time fit
fit_time, _ = bench.measure_function_time(pca.fit, X_train, params=params)

# Time transform
transform_time, _ = bench.measure_function_time(
    pca.transform, X_train, params=params)

bench.print_output(library='cuml', algorithm='pca',
                   stages=['training', 'transformation'],
                   params=params, functions=['PCA.fit', 'PCA.transform'],
                   times=[fit_time, transform_time], accuracy_type=None,
                   accuracies=[None, None], data=[X_train, X_test],
                   alg_instance=pca)
