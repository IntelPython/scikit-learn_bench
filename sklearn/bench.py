# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import sklearn

def sklearn_set_no_input_check():
    try:
        sklearn.set_config(assume_finite=True)
    except AttributeError:
        try:
            sklearn._ASSUME_FINITE = True
        except AttributeError:
            sklearn.utils.validation._assert_all_finite = lambda X: None


def set_daal_num_threads(num_threads):
    try:
        import daal4py
        if num_threads:
            daal4py.daalinit(nthreads=num_threads)
    except ImportError:
        print("@ Package 'daal4py' was not found. Number of threads is being ignored")


def prepare_benchmark(args):
    try:
        if args.num_threads > 0:
            set_daal_num_threads(args.num_threads)
        num_threads = args.num_threads
        import daal4py
        daal_version = daal4py.__daal_run_version__
    except ImportError:
        num_threads = 1
        daal_version = None

    return num_threads, daal_version


sklearn_set_no_input_check()

