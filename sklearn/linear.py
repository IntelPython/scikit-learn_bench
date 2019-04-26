# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn import linear_model
from args import getArguments, coreString
import sklearn
import bench

import argparse
argParser = argparse.ArgumentParser(prog="linear.py",
                                    description="sklearn linear regression benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = getArguments(argParser)
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
y = rand(p,n)

regr = linear_model.LinearRegression(n_jobs=int(args.num_threads))

@st_time
def test_fit(X,y):
    regr.fit(X,y)

@st_time
def test_predict(X):
    regr.predict(X)

print (','.join([args.batchID, args.arch, args.prefix, "Linear.fit", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
test_fit(X, y)
print (','.join([args.batchID, args.arch, args.prefix, "Linear.prediction", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
test_predict(Xp)
