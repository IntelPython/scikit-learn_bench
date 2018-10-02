# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import argparse
import timeit
import sys

try:
    import numpy.random_intel as nri
except ImportError:
    raise ImportError("numpy.random_intel is not found, please use Intel Distribution for Python")

def ch_size(n):
    return n * (n + 1) // 2


def get_cluster_centers(clusters, features):
    rs = nri.RandomState(1234, brng='SFMT19937')
    cluster_centers = rs.randn(clusters, features)
    cluster_centers *= np.double(clusters)
    return cluster_centers


def gen_kmeans_data(seed=None, node_id=0, clusters=3, samples=1000, features=5):
    rs = nri.RandomState(seed, brng=('MT2203', node_id))
    # generate centers
    cluster_centers = get_cluster_centers(clusters, features)
    pvec = np.full((clusters,), 1.0/clusters, dtype=np.double)
    cluster_sizes = rs.multinomial(samples, pvec)
    cluster_sizes_cum = cluster_sizes.cumsum()
    # generate clusters around those centers
    sz = 0.5
    ch = rs.uniform(low=-sz, high=sz, size=(ch_size(features),))
    data = rs.multinormal_cholesky(cluster_centers[0], ch, size = (samples,))
    diff_i0 = np.empty_like(cluster_centers[0])
    for i in range(1, clusters):
        np.subtract(cluster_centers[i], cluster_centers[0], out=diff_i0)
        data[cluster_sizes_cum[i-1]:cluster_sizes_cum[i]] += diff_i0

    j = nri.choice(range(0,samples),size=clusters, replace=False)

    return data[j], data


def generate(sx, sy, clusters, fname):
    X_init, X = gen_kmeans_data(seed=7777, node_id=0, clusters=clusters, samples=sx, features=sy) 
    print(f'Generating {fname} file ...')
    np.savetxt(fname, X, delimiter=',')
    times = []
    for n in range(10):
        t1 = timeit.default_timer()
        variances = np.var(X, axis=0)
        absTol = np.mean(variances) * 1e-16
        t2 = timeit.default_timer()
        times.append(t2-t1)
    print(f'Computing absolute threshold on this machine takes {min(times)} seconds')
    f = open(fname+'.tol', 'w')
    f.write(str(absTol))
    f.close()
    fname_init = fname + ".init"
    print(f'Generating {fname_init} file ...')
    np.savetxt(fname_init, X_init, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=str,
                         help='problem size in NxM format')
    parser.add_argument('--fname', required=True, type=str,
                         help='file name to save generated data')
    parser.add_argument('--clusters', required=True, type=int,
                         help='clusters number to generate')
    args = parser.parse_args()

    try:
        sx, sy = [int(y) for y in args.size.split('x')]
    except ValueError:
        sys.exit('bad size')

    generate(sx, sy, args.clusters, args.fname)
