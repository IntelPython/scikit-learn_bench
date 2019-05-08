# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn.metrics.pairwise import pairwise_distances
from args import getArguments, coreString

import sklearn
import bench

import argparse
argParser = argparse.ArgumentParser(prog="pairwise_distances.py",
                                    description="sklearn pairwise_distances benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        print(min(times))
        return r
    return st_func

p = args.size[0]
n = args.size[1]


X = rand(p,n)


@st_time
def cosine(X):
    cos_dist = pairwise_distances(X, metric='cosine', n_jobs=int(args.num_threads))
@st_time
def correlation(X):
    cor_dist = pairwise_distances(X, metric='correlation', n_jobs=int(args.num_threads))

print (','.join([args.batchID, args.arch, args.prefix, "Cosine", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
cosine(X)
print (','.join([args.batchID, args.arch, args.prefix, "Correlation", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
correlation(X)
