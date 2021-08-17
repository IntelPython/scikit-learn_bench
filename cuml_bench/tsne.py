import argparse
import pandas as pd
import bench
from cuml.manifold import TSNE

parser = argparse.ArgumentParser(description='cuml tsne')

parser.add_argument('--n-components', type=int, default=2,
                    help='Dimension of the embedded space.')
parser.add_argument('--early-exaggeration', type=float, default=12.0,
                    help='Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.')
parser.add_argument('--learning-rate', type=float, default=200.0,
                    help='The learning rate for t-SNE is usually in the range [10.0, 1000.0].')
parser.add_argument('--angle', type=float, default=0.5,
                    help='Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.')
parser.add_argument('--min-grad-norm', type=float, default=1e-7,
                    help='If the gradient norm is below this threshold, the optimization will be stopped.')
parser.add_argument('--random-state', type=int, default=1234)

params = bench.parse_args(parser)

# Load and convert data
X_train, X_test, _, _ = bench.load_data(params)
full_x = pd.concat([X_train, X_test])

# Create our random forest regressor
tsne = TSNE(n_components=params.n_components, early_exaggeration=params.early_exaggeration,
            learning_rate=params.learning_rate, angle=params.angle,
            min_grad_norm=params.min_grad_norm, random_state=params.random_state)

fit_time, _ = bench.measure_function_time(tsne.fit, full_x, params=params)

divergence = tsne.kl_divergence_

bench.print_output(library='cuml', algorithm='tsne',
                   stages=['training'], params=params,
                   functions=['tsne.fit'],
                   times=[fit_time], metric_type='divergence',
                   metrics=[divergence], data=[full_x],
                   alg_instance=tsne)
