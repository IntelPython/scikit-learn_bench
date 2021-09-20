import argparse
import bench
from cuml.manifold import TSNE

parser = argparse.ArgumentParser(description='cuml tsne')

parser.add_argument('--n-components', type=int, default=2,
                    help='The dimension of the embedded space.')
parser.add_argument('--early-exaggeration', type=float, default=12.0,
                    help='This factor increases the attractive forces between points '
                    'and allows points to move around more freely, '
                    'finding their nearest neighbors more easily.')
parser.add_argument('--learning-rate', type=float, default=200.0,
                    help='The learning rate for t-SNE is usually in the range [10.0, 1000.0].')
parser.add_argument('--angle', type=float, default=0.5,
                    help='Angular size. This is the trade-off between speed and accuracy.')
parser.add_argument('--min-grad-norm', type=float, default=1e-7,
                    help='If the gradient norm is below this threshold,'
                    'the optimization is stopped.')
parser.add_argument('--random-state', type=int, default=1234)
params = bench.parse_args(parser)

# Load and convert data
X, _, _, _ = bench.load_data(params)

# Create our random forest regressor
tsne = TSNE(n_components=params.n_components, early_exaggeration=params.early_exaggeration,
            learning_rate=params.learning_rate, angle=params.angle,
            min_grad_norm=params.min_grad_norm, random_state=params.random_state)

fit_time, _ = bench.measure_function_time(tsne.fit, X, params=params)
# divergence = tsne.kl_divergence_

bench.print_output(library='cuml', algorithm='tsne',
                   stages=['training'], params=params,
                   functions=['tsne.fit'],
                   times=[fit_time], metric_type=None,
                   metrics=None, data=[X],
                   alg_instance=tsne)
