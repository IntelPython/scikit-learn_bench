# Copyright (C) 2017-2018 Intel Corporation
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
sklearn._ASSUME_FINITE = True

if sklearn.__version__ == '0.18.2':
    sklearn.utils.validation._assert_all_finite = lambda X: None

import argparse
argParser = argparse.ArgumentParser(prog="kmeans.py",
                                    description="sklearn kmeans benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argParser.add_argument('--input', default=None, help='KMeans input file')
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
            if sum(times) > 60:
                break
        print (min(times), end='')
        return r
    return st_func

problem_sizes = [
        (args.size[0], args.size[1])]

if args.input is None:
    fname = f'../kmeans_{args.size[0]}x{args.size[1]}.csv'
elif os.path.isdir(args.input):
    fname = os.path.join(args.input, f'kmeans_{args.size[0]}x{args.size[1]}.csv')
else:
    fname = args.input

X = np.loadtxt(fname, dtype=np.float64, delimiter=',')
X_init = np.loadtxt(fname + ".init", dtype=np.float64, delimiter=',')

kmeans = KMeans(n_clusters=10, n_jobs=int(args.core_number), tol=1e-16, max_iter=100, n_init=1, init=X_init)
@st_time
def train(X):
    kmeans.fit(X)

for p, n in problem_sizes:
    print (','.join([args.batchID, args.arch, args.prefix, "KMeans.fit", coreString(args.core_number), "Double", "%sx%s" % (p,n)]), end=',')
    X_local = X
    train(X_local)
    print('')
