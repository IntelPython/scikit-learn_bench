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
    from sklearn.metrics.pairwise import pairwise_distances

    # Load data
    X, _, _, _ = bench.load_data(params, generated_data=['X_train'], add_dtype=True)

    time, _ = bench.measure_function_time(pairwise_distances, X, metric=params.metric,
                                          n_jobs=params.n_jobs, params=params)

    bench.print_output(library='sklearn', algorithm='distances', stages=['computation'],
                       params=params, functions=[params.metric.capitalize()],
                       times=[time], metric_type=None, metrics=[None], data=[X],
                       alg_params={'metric': params.metric})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                     'benchmark')
    parser.add_argument('--metric', default='cosine',
                        choices=['cosine', 'correlation'],
                        help='Metric to test for pairwise distances')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
