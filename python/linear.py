# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from sklearn import linear_model
from args import getArguments, coreString
import sklearn
sklearn._ASSUME_FINITE = True
if sklearn.__version__ == '0.18.2':
    sklearn.utils.validation._assert_all_finite = lambda X: None

import argparse
argParser = argparse.ArgumentParser(prog="linear.py",
                                    description="sklearn linear regression benchmark",
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
Xp = rand(p,n)
y = rand(p,n)

regr = linear_model.LinearRegression(n_jobs=int(args.core_number))

@st_time
def test_fit(X,y):
    regr.fit(X,y)

@st_time
def test_predict(X):
    regr.predict(X)

print (','.join([args.batchID, args.arch, args.prefix, "Linear.fit", coreString(args.core_number), "Double", "%sx%s" % (p,n)]), end=',')
test_fit(X, y)
print (','.join([args.batchID, args.arch, args.prefix, "Linear.prediction", coreString(args.core_number), "Double", "%sx%s" % (p,n)]), end=',')
test_predict(Xp)
