# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
import numpy as np
import sklearn
import timeit


def _parse_size(string, dim=2):
    try:
        tup = tuple(int(n) for n in string.replace('x', ',').split(','))
    except Exception as e:
        msg = (
            f"Invalid size '{string}': sizes must be integers separated by "
            f"'x' or ','."
        )
        raise argparse.ArgumentTypeError(msg) from e

    if len(tup) != dim:
        msg = f'Expected size parameter of {dim} dimensions but got {len(tup)}'
        raise argparse.ArgumentTypeError(msg)

    return tup


def parse_args(parser, size=None, dtypes=None, loop_types=(),
               n_jobs_supported=False):
    '''
    Add common arguments useful for most benchmarks and parse.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the arguments should be added.
    size : tuple of int, optional
        Enable '--size' argument with this default size.
        If None (default), no '--size' argument will be added.
    dtypes : iterable of str, optional
        Enable '--dtype' argument with the given acceptable dtypes.
        The first one is used as the default.
    loop_types : iterable of str, optional
        Add arguments like '--fit-inner-loops' and '--fit-outer-loops',
        useful for tweaking runtime of the benchmark.
    n_jobs_supported : bool
        If set to True, generate a n_jobs member in the argparse Namespace
        corresponding to the optimal n_jobs parameter for scikit-learn.
        Otherwise, n_jobs will be set to None.

    Returns
    -------
    parser : argparse.ArgumentParser
        Parser to which the arguments were added.
        This is the same parser that was passed to this function.
    '''

    parser.add_argument('-n', '--num-threads', '--core-number', default=-1,
                        dest='threads', type=int,
                        help='Number of threads to use')
    parser.add_argument('-a', '--arch', default='?',
                        help='Machine architecture, for bookkeeping')
    parser.add_argument('-b', '--batch', '--batchID', default='?',
                        help='Batch ID, for bookkeeping')
    parser.add_argument('-p', '--prefix', default='sklearn',
                        help='Prefix string, for bookkeeping')
    parser.add_argument('--header', default=False, action='store_true',
                        help='Output CSV header')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Output extra debug messages')

    if dtypes is not None:
        parser.add_argument('-d', '--dtype', default=np.dtype(dtypes[0]),
                            type=np.dtype, choices=dtypes,
                            help='Data type to use (see numpy.dtype docs)')

    if size is not None:
        parser.add_argument('-s', '--size', default=size, type=_parse_size,
                            dest='shape',
                            help="Problem size, delimited by 'x' or ','")

    if len(loop_types) == 0:
        loop_types = (None,)

    for loop in loop_types:

        if loop is not None:
            loop_dash = f'{loop}-'
            loop_for = f'for {loop} '
        else:
            loop_dash = loop_for = ''

        parser.add_argument(f'--{loop_dash}inner-loops', default=3, type=int,
                            help=f'Number of inner loop iterations {loop_for}'
                                 f'(we take the mean over inner iterations)')
        parser.add_argument(f'--{loop_dash}outer-loops', default=1, type=int,
                            help=f'Number of outer loop iterations {loop_for}'
                                 f'(we take the min over outer iterations)')

    params = parser.parse_args()

    # Ask DAAL what it thinks about this number of threads
    num_threads, daal_version = prepare_daal(num_threads=params.threads)
    if params.verbose and daal_version:
        print(f'@ Found DAAL version {daal_version}')
        print(f'@ DAAL gave us {num_threads} threads')

    n_jobs = None
    if n_jobs_supported and not daal_version:
        n_jobs = num_threads = params.num_threads

    # Set threading and DAAL related params here
    setattr(params, 'threads', num_threads)
    setattr(params, 'daal_version', daal_version)
    setattr(params, 'using_daal', daal_version is not None)
    setattr(params, 'n_jobs', n_jobs)

    # Set size string parameter for easy printing
    if size is not None:
        setattr(params, 'size', size_str(params.shape))

    # Very verbose output
    if params.verbose:
        print(f'@ params = {params.__dict__}')

    return params


def size_str(shape):
    return 'x'.join(str(d) for d in shape)


def print_header(columns, params):
    if params.header:
        print(','.join(columns))


def print_row(columns, params, **kwargs):
    values = []

    for col in columns:
        if col in kwargs:
            values.append(str(kwargs[col]))
        elif hasattr(params, col):
            values.append(str(getattr(params, col)))
        else:
            values.append('')

    print(','.join(values))


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
        print("@ Package 'daal4py' was not found. Number of threads "
              "is being ignored")


def prepare_daal(num_threads=-1):
    try:
        if num_threads > 0:
            set_daal_num_threads(num_threads)
        import daal4py
        num_threads = daal4py.num_threads()
        daal_version = daal4py.__daal_run_version__
    except ImportError:
        num_threads = 1
        daal_version = None

    return num_threads, daal_version


def time_mean_min(func, *args, inner_loops=1, outer_loops=1, **kwargs):
    '''
    Time the given function (inner_loops * outer_loops) times, returning the
    min of the inner loop means.

    Parameters
    ----------
    func : callable f(*args, **kwargs)
        The function to time.
    inner_loops : int
        Number of inner loop iterations to take the mean over.
    outer_loops : int
        Number of outer loop iterations to take the min over.

    Returns
    -------
    time : float
        The min of means.
    val : return value of func
        The last value returned by func.
    '''

    assert inner_loops * outer_loops > 0, \
        'Must time the function at least once'

    times = np.zeros(outer_loops, dtype='f8')

    for i in range(outer_loops):

        t0 = timeit.default_timer()
        for _ in range(inner_loops):
            val = func(*args, **kwargs)
        t1 = timeit.default_timer()

        times[i] = t1 - t0

    # We take the mean of inner loop times
    times /= inner_loops

    # We take the min of outer loop times
    return np.min(times), val


sklearn_set_no_input_check()
