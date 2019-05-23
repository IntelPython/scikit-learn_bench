# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
import sklearn
from sklearn.decomposition import PCA
from args import getArguments, coreString
import bench

import argparse
parser = argparse.ArgumentParser(description="sklearn PCA benchmark")
parser.add_argument('--svd-solver', type=str, choices=['daal', 'full'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
args = getArguments(parser)
REP = args.iteration if args.iteration != '?' else 10
core_number, daal_version = bench.prepare_benchmark(args)


def st_time(func):
    def st_func(*args, **keyArgs):
        times = []
        for n in range(REP):
            t1 = timeit.default_timer()
            r = func(*args, **keyArgs)
            t2 = timeit.default_timer()
            times.append(t2-t1)
        print(min(times))
        return r
    return st_func

p = args.size[0]
n = args.size[1]
X = rand(p,n)
Xp = rand(p,n)

if args.n_components:
    n_components = args.n_components
else:
    n_components = min((n, (2 + min((n, p))) // 3))

pca = PCA(svd_solver=args.svd_solver, whiten=args.whiten,
          n_components=args.n_components)

@st_time
def test_fit(X):
    pca.fit(X)

@st_time
def test_transform(X):
    pca.transform(X)

print (','.join([args.batchID, args.arch, args.prefix, "PCA.fit", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
test_fit(X)
print (','.join([args.batchID, args.arch, args.prefix, "PCA.transform", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
test_transform(Xp)
