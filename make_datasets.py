import argparse
import sys

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state


def gen_regression(args):
    rs = check_random_state(args.seed)
    X, y = make_regression(n_targets=1, n_samples=args.samples,
                           n_features=args.features,
                           n_informative=args.features,
                           bias=rs.normal(0, 3),
                           random_state=rs)

    if args.test_samples != 0:
        train_samples = args.samples - args.test_samples
        np.save(args.filex, X[:train_samples])
        np.save(args.filey, y[:train_samples])
        np.save(args.filextest, X[train_samples:])
        np.save(args.fileytest, y[train_samples:])
    else:
        np.save(args.filex, X)
        np.save(args.filey, y)
    return 0


def gen_classification(args):

    X, y = make_classification(n_samples=args.samples,
                               n_features=args.features,
                               n_informative=args.features,
                               n_repeated=0,
                               n_redundant=0,
                               n_classes=args.classes,
                               random_state=args.seed)
    if args.test_samples != 0:
        train_samples = args.samples - args.test_samples
        np.save(args.filex, X[:train_samples])
        np.save(args.filey, y[:train_samples])
        np.save(args.filextest, X[train_samples:])
        np.save(args.fileytest, y[train_samples:])
    else:
        np.save(args.filex, X)
        np.save(args.filey, y)
    return 0


def _ch_size(n):
    return n * (n + 1) // 2


def _get_cluster_centers(clusters, features):
    import numpy.random_intel as nri
    rs = nri.RandomState(1234, brng='SFMT19937')
    cluster_centers = rs.randn(clusters, features)
    cluster_centers *= np.double(clusters)
    return cluster_centers


def gen_kmeans(args):
    try:
        import numpy.random_intel as nri
    except ImportError:
        raise ImportError('numpy.random_intel not found. '
                          'Please use Intel Distribution for Python.')


    rs = nri.RandomState(args.seed, brng=('MT2203', args.node_id))

    # generate centers
    cluster_centers = _get_cluster_centers(args.clusters, args.features)
    pvec = np.full((args.clusters,), 1.0 / args.clusters, dtype=np.double)
    cluster_sizes = rs.multinomial(args.samples, pvec)
    cluster_sizes_cum = cluster_sizes.cumsum()

    # generate clusters around those centers
    sz = 0.5
    ch = rs.uniform(low=-sz, high=sz, size=(_ch_size(args.features),))
    data = rs.multinormal_cholesky(cluster_centers[0], ch,
                                   size=(args.samples,))
    diff_i0 = np.empty_like(cluster_centers[0])
    for i in range(1, args.clusters):
        np.subtract(cluster_centers[i], cluster_centers[0], out=diff_i0)
        data[cluster_sizes_cum[i-1]:cluster_sizes_cum[i]] += diff_i0

    j = nri.choice(range(0, args.samples), size=args.clusters, replace=False)

    X_init = data[j]
    X = data
    times = []
    import timeit
    for n in range(10):
        t1 = timeit.default_timer()
        variances = np.var(X, axis=0)
        absTol = np.mean(variances) * 1e-16
        t2 = timeit.default_timer()
        times.append(t2-t1)
    print(f'Computing absolute threshold on this machine '
          f'takes {min(times)} seconds')

    if args.test_samples != 0:
        train_samples = args.samples - args.test_samples
        np.save(args.filex, X[:train_samples])
        np.save(args.filextest, X[train_samples:])
    else:
        np.save(args.filex, X)
    np.save(args.filei, X_init)
    np.save(args.filet, absTol)
    return 0


def main():

    parser = argparse.ArgumentParser(
            description='Dataset generator using scikit-learn')
    parser.add_argument('-f', '--features', type=int, default=1000,
                        help='Number of features in dataset')
    parser.add_argument('-s', '--samples', type=int, default=10000,
                        help='Number of samples in dataset')
    parser.add_argument('-ts', '--test-samples', type=int, default=0,
                        help='Number of test samples in dataset')
    parser.add_argument('-d', '--seed', type=int, default=0,
                        help='Seed for random state')
    subparsers = parser.add_subparsers(dest='problem')
    subparsers.required = True

    regr_parser = subparsers.add_parser('regression',
                                        help='Regression data')
    regr_parser.set_defaults(func=gen_regression)
    regr_parser.add_argument('-x', '--filex', '--fileX', type=str,
                             required=True, help='Path to save matrix X')
    regr_parser.add_argument('-y', '--filey', '--fileY', type=str,
                             required=True, help='Path to save vector y')
    regr_parser.add_argument('-xt', '--filextest', '--fileXtest', type=str,
                             help='Path to save test matrix X')
    regr_parser.add_argument('-yt', '--fileytest', '--fileYtest', type=str,
                             help='Path to save test vector y')

    clsf_parser = subparsers.add_parser('classification',
                                        help='Classification data')
    clsf_parser.set_defaults(func=gen_classification)
    clsf_parser.add_argument('-c', '--classes', type=int, default=5,
                             help='Number of classes')
    clsf_parser.add_argument('-x', '--filex', '--fileX', type=str,
                             required=True, help='Path to save matrix X')
    clsf_parser.add_argument('-y', '--filey', '--fileY', type=str,
                             required=True,
                             help='Path to save label vector y')
    clsf_parser.add_argument('-xt', '--filextest', '--fileXtest', type=str,
                             help='Path to save test matrix X')
    clsf_parser.add_argument('-yt', '--fileytest', '--fileYtest', type=str,
                             help='Path to save test vector y')

    kmeans_parser = subparsers.add_parser('kmeans',
                                          help='KMeans clustering data')
    kmeans_parser.set_defaults(func=gen_kmeans)
    kmeans_parser.add_argument('-c', '--clusters', type=int, default=10,
                               help='Number of clusters to generate')
    kmeans_parser.add_argument('-n', '--node-id', type=int, default=0,
                               help='ID of member of MKL BRNG')
    kmeans_parser.add_argument('-x', '--filex', '--fileX', type=str,
                               required=True, help='Path to save matrix X')
    kmeans_parser.add_argument('-xt', '--filextest', '--fileXtest', type=str,
                               help='Path to test save matrix X')
    kmeans_parser.add_argument('-i', '--filei', '--fileI', type=str,
                               required=True,
                               help='Path to save initial cluster centers')
    kmeans_parser.add_argument('-t', '--filet', '--fileT', type=str,
                               required=True,
                               help='Path to save absolute threshold')

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
