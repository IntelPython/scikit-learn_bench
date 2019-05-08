# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from daal4py import correlation_distance, cosine_distance, daalinit
from args import getArguments, coreString
from bench import prepare_benchmark

import argparse
argParser = argparse.ArgumentParser(prog="pairwise_distances.py",
                                    description="sklearn pairwise_distances benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = getArguments(argParser)
REP = args.iteration if args.iteration != '?' else 10
core_number, daal_version = prepare_benchmark(args)


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
    cos_dist = cosine_distance().compute(X)
@st_time
def correlation(X):
    cor_dist = correlation_distance().compute(X)

print (','.join([args.batchID, args.arch, args.prefix, "Cosine", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
cosine(X)
print (','.join([args.batchID, args.arch, args.prefix, "Correlation", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
correlation(X)
