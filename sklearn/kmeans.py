# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import os
import timeit
from numpy.random import rand
from sklearn.cluster import KMeans
from args import getArguments,coreString
import sklearn
import bench

import argparse
argParser = argparse.ArgumentParser(prog="kmeans.py",
                                    description="sklearn kmeans benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argParser.add_argument('-x', '--filex', '--fileX', '--input',
                       type=str, help='Points to cluster')
argParser.add_argument('-i', '--filei', '--fileI', '--init',
                       type=str, help='Initial clusters')
# argParser.add_argument('-t', '--filet', '--fileT', '--tol',
#                        type=str, help='Absolute threshold')
argParser.add_argument('-m', '--data-multiplier', default=100,
                       type=int, help='Data multiplier')
args = getArguments(argParser)
REP = args.iteration if args.iteration != '?' else 10
num_threads, daal_version = bench.prepare_benchmark(args)


def st_time(func):
    def st_func(*args, **keyArgs):
        times = []
        for n in range(REP):
            t1 = timeit.default_timer()
            r = func(*args, **keyArgs)
            t2 = timeit.default_timer()
            times.append(t2-t1)
            if sum(times) > 60:
                break
        print (min(times), end='')
        return r
    return st_func

problem_sizes = [(args.size[0], args.size[1])]

X = np.load(args.filex)
X_init = np.load(args.filei)
X_mult = np.vstack((X,) * args.data_multiplier)

kmeans = KMeans(n_clusters=10, n_jobs=int(args.num_threads), tol=1e-16,
                max_iter=100, n_init=1, init=X_init)
@st_time
def train(X):
    kmeans.fit(X)


@st_time
def predict(X):
    kmeans.predict(X)


for p, n in problem_sizes:
    print(','.join([args.batchID, args.arch, args.prefix, "KMeans.fit",
                    coreString(args.num_threads), "Double", "%sx%s" % (p,n)]),
          end=',')
    X_local = X
    train(X_local)
    print('')

    print(','.join([args.batchID, args.arch, args.prefix, "KMeans.predict",
                    coreString(args.num_threads), "Double", "%sx%s" % (p,n)]),
          end=',')
    predict(X_mult)
    print('')
