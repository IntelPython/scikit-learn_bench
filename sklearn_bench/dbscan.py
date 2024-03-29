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


def main():
    from sklearn.cluster import DBSCAN

    # Load generated data
    X, _, _, _ = bench.load_data(params, add_dtype=True)

    # Create our clustering object
    dbscan = DBSCAN(eps=params.eps, n_jobs=params.n_jobs,
                    min_samples=params.min_samples, metric='euclidean',
                    algorithm='auto')

    # N.B. algorithm='auto' will select oneAPI Data Analytics Library (oneDAL)
    # brute force method when running daal4py-patched scikit-learn, and probably
    #  'kdtree' when running unpatched scikit-learn.

    # Time fit
    time, _ = bench.measure_function_time(dbscan.fit, X, params=params)
    labels = dbscan.labels_

    params.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    acc = bench.davies_bouldin_score(X, labels)

    bench.print_output(library='sklearn', algorithm='dbscan', stages=['training'],
                       params=params, functions=['DBSCAN'], times=[time],
                       metrics=[acc], metric_type='davies_bouldin_score',
                       data=[X], alg_instance=dbscan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn DBSCAN benchmark')
    parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10.,
                        help='Radius of neighborhood of a point')
    parser.add_argument('-m', '--min-samples', default=5, type=int,
                        help='The minimum number of samples required in a '
                        'neighborhood to consider a point a core point')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
