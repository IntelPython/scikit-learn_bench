# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn.metrics.pairwise import pairwise_distances
from args import getArguments, coreString

import sklearn
sklearn._ASSUME_FINITE = True

if sklearn.__version__ == '0.18.2':
    sklearn.utils.validation._assert_all_finite = lambda X: None

import argparse
argParser = argparse.ArgumentParser(prog="pairwise_distances.py",
                                    description="sklearn pairwise_distances benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = getArguments(argParser)
REP = args.iteration if args.iteration != '?' else 10
try:
    from daal.services import Environment
    nThreadsInit = Environment.getInstance().getNumberOfThreads()
    core_number = int(args.core_number)
    if core_number != -1:
        Environment.getInstance().setNumberOfThreads(core_number)
except:
    pass


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
    cos_dist = pairwise_distances(X, metric='cosine', n_jobs=int(args.core_number))
@st_time
def correlation(X):
    cor_dist = pairwise_distances(X, metric='correlation', n_jobs=int(args.core_number))

print (','.join([args.batchID, args.arch, args.prefix, "Cosine", coreString(args.core_number), "Double", "%sx%s" % (p,n)]), end=',')
cosine(X)
print (','.join([args.batchID, args.arch, args.prefix, "Correlation", coreString(args.core_number), "Double", "%sx%s" % (p,n)]), end=',')
correlation(X)
