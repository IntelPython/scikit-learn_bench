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
    from sklearn.manifold import TSNE

    # Load and convert data
    X, _, _, _ = bench.load_data(params)

    # Create our TSNE model
    tsne = TSNE(n_components=params.n_components, early_exaggeration=params.early_exaggeration,
                learning_rate=params.learning_rate, angle=params.angle,
                min_grad_norm=params.min_grad_norm, random_state=params.random_state)

    fit_time, _ = bench.measure_function_time(tsne.fit, X, params=params)
    divergence = tsne.kl_divergence_

    bench.print_output(
        library='sklearn',
        algorithm='TSNE',
        stages=['training'],
        params=params,
        functions=['TSNE.fit'],
        times=[fit_time],
        metric_type='divergence',
        metrics=[divergence],
        data=[X],
        alg_instance=tsne,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn tsne '
                                     'regression benchmark')

    parser.add_argument('--n-components', type=int, default=2,
                        help='Dimension of the embedded space.')
    parser.add_argument('--early-exaggeration', type=float, default=12.0,
                        help='Controls how tight natural clusters in the'
                        'original space are in the embedded space and how '
                        'much space will be between them.')
    parser.add_argument('--learning-rate', type=float, default=200.0,
                        help='The learning rate for t-SNE is usually in the range [10.0, 1000.0].')
    parser.add_argument('--angle', type=float, default=0.5,
                        help='Only used if method=’barnes_hut’'
                        'This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.')
    parser.add_argument('--min-grad-norm', type=float, default=1e-7,
                        help='If the gradient norm is below this threshold,'
                        'the optimization will be stopped.')
    parser.add_argument('--random-state', type=int, default=1234)

    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
