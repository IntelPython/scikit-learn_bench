# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
import numpy as np
import sklearn
import timeit
import json


def get_dtype(data):
    '''
    Get type of input data as numpy.dtype
    '''
    if hasattr(data, 'dtype'):
        return data.dtype
    elif hasattr(data, 'dtypes'):
        return str(data.dtypes[0])
    elif hasattr(data, 'values'):
        return data.values.dtype
    else:
        raise ValueError(f'Impossible to get data type of {type(data)}')


try:
    from daal4py.sklearn._utils import getFPType
except ImportError:
    def getFPType(X):
        dtype = str(get_dtype(X))
        if 'float32' in dtype:
            return 'float'
        elif 'float64' in dtype:
            return 'double'
        else:
            ValueError('Unknown type')


def sklearn_disable_finiteness_check():
    try:
        sklearn.set_config(assume_finite=True)
    except AttributeError:
        try:
            sklearn._ASSUME_FINITE = True
        except AttributeError:
            sklearn.utils.validation._assert_all_finite = lambda X: None


def _parse_size(string, dim=2):
    try:
        tup = tuple(int(n) for n in string.replace('x', ',').split(','))
    except Exception as e:
        msg = (
            f'Invalid size "{string}": sizes must be integers separated by '
            f'"x" or ",".'
        )
        raise argparse.ArgumentTypeError(msg) from e

    if len(tup) != dim:
        msg = f'Expected size parameter of {dim} dimensions but got {len(tup)}'
        raise argparse.ArgumentTypeError(msg)

    return tup


def float_or_int(string):
    if '.' in string:
        return float(string)
    else:
        return int(string)


def parse_args(parser, size=None, loop_types=(),
               n_jobs_supported=False, prefix='sklearn'):
    '''
    Add common arguments useful for most benchmarks and parse.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the arguments should be added.
    size : tuple of int, optional
        Enable '--size' argument with this default size.
        If None (default), no '--size' argument will be added.
    loop_types : iterable of str, optional
        Add arguments like '--fit-inner-loops' and '--fit-outer-loops',
        useful for tweaking runtime of the benchmark.
    n_jobs_supported : bool
        If set to True, generate a n_jobs member in the argparse Namespace
        corresponding to the optimal n_jobs parameter for scikit-learn.
        Otherwise, n_jobs will be set to None.
    prefix : str, optional, default 'sklearn'
        The default prefix to report

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
    parser.add_argument('-p', '--prefix', default=prefix,
                        help='Prefix string, for bookkeeping')
    parser.add_argument('--header', default=False, action='store_true',
                        help='Output CSV header')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Output extra debug messages')
    parser.add_argument('--data-format', type=str, default='numpy',
                        choices=('numpy', 'pandas', 'cudf'),
                        help='Data format: numpy (default), pandas, cudf')
    parser.add_argument('--data-order', type=str, default='C',
                        choices=('C', 'F'),
                        help='Data order: C (row-major, default) or'
                             'F (column-major)')
    parser.add_argument('-d', '--dtype', type=np.dtype, default=np.float64,
                        choices=(np.float32, np.float64),
                        help='Data type: float64 (default) or float32')
    parser.add_argument('--check-finiteness', default=False,
                        action='store_true',
                        help='Check finiteness in sklearn input check'
                             '(disabled by default)')
    parser.add_argument('--output-format', type=str, default='csv',
                        choices=('csv', 'json'),
                        help='Output format: csv (default) or json')
    parser.add_argument('--time-method', type=str, default='mean_min',
                        choices=('box_filter', 'mean_min'),
                        help='Method used for time mesurements')
    parser.add_argument('--box-filter-measurements', type=int, default=100,
                        help='Maximum number of measurements in box filter')
    parser.add_argument('--inner-loops', default=100, type=int,
                        help='Maximum inner loop iterations '
                             '(we take the mean over inner iterations)')
    parser.add_argument('--outer-loops', default=100, type=int,
                        help='Maximum outer loop iterations '
                             '(we take the min over outer iterations)')
    parser.add_argument('--time-limit', default=10., type=float,
                        help='Target time to spend to benchmark')
    parser.add_argument('--goal-outer-loops', default=10,
                        type=int, dest='goal',
                        help='Number of outer loops to aim '
                             'while automatically picking number of '
                             'inner loops. If zero, do not automatically '
                             'decide number of inner loops.')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Seed to pass as random_state')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name')

    for data in ['X', 'y']:
        for stage in ['train', 'test']:
            parser.add_argument(f'--file-{data}-{stage}',
                                type=argparse.FileType('r'),
                                help=f'Input file with {data}_{stage},'
                                     'in NPY format')

    if size is not None:
        parser.add_argument('-s', '--size', default=size, type=_parse_size,
                            dest='shape',
                            help='Problem size, delimited by "x" or ","')

    params = parser.parse_args()

    # disable finiteness check (default)
    if not params.check_finiteness:
        sklearn_disable_finiteness_check()

    # Ask DAAL what it thinks about this number of threads
    num_threads, daal_version = prepare_daal(num_threads=params.threads)
    if params.verbose and daal_version:
        print(f'@ Found DAAL version {daal_version}')
        print(f'@ DAAL gave us {num_threads} threads')

    n_jobs = None
    if n_jobs_supported and not daal_version:
        n_jobs = num_threads = params.threads

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


def set_daal_num_threads(num_threads):
    try:
        import daal4py
        if num_threads:
            daal4py.daalinit(nthreads=num_threads)
    except ImportError:
        print('@ Package "daal4py" was not found. Number of threads '
              'is being ignored')


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


def measure_function_time(func, *args, params, **kwargs):
    if params.time_method == 'mean_min':
        return time_mean_min(func, *args,
                             outer_loops=params.outer_loops,
                             inner_loops=params.inner_loops,
                             goal_outer_loops=params.goal,
                             time_limit=params.time_limit,
                             verbose=params.verbose, **kwargs)
    else:
        return time_box_filter(func, *args,
                               n_meas=params.box_filter_measurements,
                               time_limit=params.time_limit, **kwargs)


def time_box_filter(func, *args, n_meas, time_limit, **kwargs):
    times = []
    while len(times) < n_meas:
        t0 = timeit.default_timer()
        val = func(*args, **kwargs)
        t1 = timeit.default_timer()
        print(t1-t0)
        times.append(t1-t0)
        if sum(times) > time_limit:
            break

    def box_filter(timing, left=0.25, right=0.75):
        timing.sort()
        size = len(timing)
        if size == 1:
            return timing[0]
        Q1, Q2 = timing[int(size * left)], timing[int(size * right)]
        IQ = Q2 - Q1
        lower = Q1 - 1.5 * IQ
        upper = Q2 + 1.5 * IQ
        result = np.array([item for item in timing if lower < item < upper])
        return np.mean(result)

    return box_filter(times), val


def time_mean_min(func, *args, inner_loops=1, outer_loops=1, time_limit=10.,
                  goal_outer_loops=10, verbose=False, **kwargs):
    '''
    Time the given function (inner_loops * outer_loops) times, returning the
    min of the inner loop means.

    Parameters
    ----------
    func : callable f(*args, **kwargs)
        The function to time.
    inner_loops : int
        Maximum number of inner loop iterations to take the mean over.
    outer_loops : int
        Maximum number of outer loop iterations to take the min over.
    time_limit : double
        Number of seconds to aim for. If accumulated time exceeds time_limit
        in outer loops, exit without running more outer loops. If zero,
        disable time limit.
    goal_outer_loops : int
        Number of outer loop iterations to aim for by taking warmup rounds
        and tuning inner_loops automatically.
    verbose : boolean
        If True, print outer loop timings and miscellaneous information.

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
    total_time = 0.

    # Warm-up iterations to determine optimal inner_loops
    warmup = (goal_outer_loops > 0)
    warmup_time = 0.
    last_warmup = 0.
    if warmup:
        for _ in range(inner_loops):
            t0 = timeit.default_timer()
            val = func(*args, **kwargs)
            t1 = timeit.default_timer()

            last_warmup = t1 - t0
            warmup_time += last_warmup
            if warmup_time > time_limit / 10:
                break

        inner_loops = max(1, int(time_limit / last_warmup / goal_outer_loops))
        logverbose(f'Optimal inner loops = {inner_loops}', verbose)

    if last_warmup > time_limit:
        # If we took too much time in warm-up, just use those numbers
        logverbose(f'A single warmup iteration took {last_warmup:0.2f}s '
                   f'> {time_limit:0.2f}s - not performing any more timings',
                   verbose)
        outer_loops = 1
        inner_loops = 1
        times[0] = last_warmup
        times = times[:1]
    else:
        # Otherwise, actually take the timing
        for i in range(outer_loops):

            t0 = timeit.default_timer()
            for _ in range(inner_loops):
                val = func(*args, **kwargs)
            t1 = timeit.default_timer()

            times[i] = t1 - t0
            total_time += times[i]

            if time_limit > 0 and total_time > time_limit:
                logverbose(f'TT={total_time:0.2f}s exceeding {time_limit}s '
                           f'after iteration {i+1}', verbose)
                outer_loops = i + 1
                times = times[:outer_loops]
                break

    # We take the mean of inner loop times
    times /= inner_loops
    logverbose('Mean times [s]', verbose)
    logverbose(f'{times}', verbose)

    # We take the min of outer loop times
    return np.min(times), val


def logverbose(msg, verbose):
    '''
    Print msg as a verbose logging message only if verbose is True
    '''
    if verbose:
        print('@', msg)


def convert_to_numpy(data):
    '''
    Convert input data to numpy array
    '''
    if 'cudf' in str(type(data)):
        data = data.to_pandas().values
    elif 'pandas' in str(type(data)):
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    elif 'numba.cuda.cudadrv.devicearray.DeviceNDArray' in str(type(data)):
        data = np.array(data)
    else:
        raise TypeError(
            f'Unknown data format "{type(data)}" for convertion to np.ndarray')
    return data


def columnwise_score(y, yp, score_func):
    y = convert_to_numpy(y)
    yp = convert_to_numpy(yp)
    if y.ndim + yp.ndim > 2:
        if 1 in (y.shape + yp.shape)[1:]:
            if y.ndim > 1:
                y = y[:, 0]
            if yp.ndim > 1:
                yp = yp[:, 0]
        else:
            return [score_func(y[i], yp[i]) for i in range(y.shape[1])]
    return score_func(y, yp)


def accuracy_score(y, yp):
    return columnwise_score(y, yp, lambda y1, y2: np.mean(y1 == y2))


def rmse_score(y, yp):
    return columnwise_score(
        y, yp, lambda y1, y2: float(np.sqrt(np.mean((y1 - y2)**2))))


def convert_data(data, dtype, data_order, data_format):
    '''
    Convert input data (numpy array) to needed format, type and order
    '''
    # Firstly, change order and type of data
    if data_order == 'F':
        data = np.asfortranarray(data, dtype)
    elif data_order == 'C':
        data = np.ascontiguousarray(data, dtype)

    # Secondly, change format of data
    if data_format == 'numpy':
        return data
    elif data_format == 'pandas':
        import pandas as pd

        if data.ndim == 1:
            return pd.Series(data)
        else:
            return pd.DataFrame(data)
    elif data_format == 'cudf':
        import cudf
        import pandas as pd

        return cudf.DataFrame.from_pandas(pd.DataFrame(data))


def read_csv(filename, params):
    from string import ascii_lowercase, ascii_uppercase

    # find out header existance
    header_letters = set(
        ascii_lowercase.replace('e', '') + ascii_uppercase.replace('E', ''))
    with open(filename, 'r') as file:
        first_line = file.readline()
        while 'nan' in first_line:
            first_line = first_line.replace('nan', '')
        header = 0 if len(header_letters & set(first_line)) != 0 else None
    # try to read csv with pandas and fall back to numpy reader if failed
    try:
        import pandas as pd
        data = pd.read_csv(filename, header=header, dtype=params.dtype).values
    except ImportError:
        data = np.genfromtxt(filename, delimiter=',', dtype=params.dtype,
                             skip_header=0 if header is None else 1)

    if data.ndim == 2:
        if data.shape[1] == 1:
            data = data.reshape((data.shape[0],))

    return data


def load_data(params, generated_data=[], add_dtype=False, label_2d=False,
              int_label=False):
    full_data = {
        file: None for file in ['X_train', 'X_test', 'y_train', 'y_test']
    }
    param_vars = vars(params)
    int_dtype = np.int32 if '32' in str(params.dtype) else np.int64
    for element in full_data:
        file_arg = f'file_{element}'
        # load and convert data from npy/csv file if path is specified
        if param_vars[file_arg] is not None:
            if param_vars[file_arg].name.endswith('.npy'):
                data = np.load(param_vars[file_arg].name)
            else:
                data = read_csv(param_vars[file_arg].name, params)
            full_data[element] = convert_data(
                data,
                int_dtype if 'y' in element and int_label else params.dtype,
                params.data_order, params.data_format
            )
        # generate and convert data if it's marked and path isn't specified
        if full_data[element] is None and element in generated_data:
            full_data[element] = convert_data(
                np.random.rand(*params.shape),
                int_dtype if 'y' in element and int_label else params.dtype,
                params.data_order, params.data_format)
        # convert existing labels from 1- to 2-dimensional
        # if it's forced and possible
        if full_data[element] is not None and 'y' in element and label_2d and hasattr(full_data[element], 'reshape'):
            full_data[element] = full_data[element].reshape(
                (full_data[element].shape[0], 1))
        # add dtype property to data if it's needed and doesn't exist
        if full_data[element] is not None and add_dtype and not hasattr(full_data[element], 'dtype'):
            if hasattr(full_data[element], 'values'):
                full_data[element].dtype = full_data[element].values.dtype
            elif hasattr(full_data[element], 'dtypes'):
                full_data[element].dtype = full_data[element].dtypes[0].type

    params.dtype = get_dtype(full_data['X_train'])
    # add size to parameters which is need for some cases
    if not hasattr(params, 'size'):
        params.size = size_str(full_data['X_train'].shape)

    # clone train data to test if test data is None
    for data in ['X', 'y']:
        if full_data[f'{data}_train'] is not None and full_data[f'{data}_test'] is None:
            full_data[f'{data}_test'] = full_data[f'{data}_train']
    return tuple(full_data.values())


def output_csv(columns, params, functions, times, accuracies=None):
    print_header(columns, params)
    if accuracies is None:
        accuracies = [None]*len(functions)
    for i in range(len(functions)):
        if accuracies[i] is not None:
            print_row(columns, params, function=functions[i], time=times[i],
                      accuracy=accuracies[i])
        else:
            print_row(columns, params, function=functions[i], time=times[i])


def gen_basic_dict(library, algorithm, stage, params, data, alg_instance=None,
                   alg_params=None):
    result = {
        'library': library,
        'algorithm': algorithm,
        'stage': stage,
        'input_data': {
            'data_format': params.data_format,
            'data_order': params.data_order,
            'data_type': str(params.dtype),
            'dataset_name': params.dataset_name,
            'rows': data.shape[0],
            'columns': data.shape[1]
        }
    }
    result['algorithm_parameters'] = {}
    if alg_instance is not None:
        if 'Booster' in str(type(alg_instance)):
            alg_instance_params = dict(alg_instance.attributes())
        else:
            alg_instance_params = dict(alg_instance.get_params())
        result['algorithm_parameters'].update(alg_instance_params)
    if alg_params is not None:
        result['algorithm_parameters'].update(alg_params)
    return result


def print_output(library, algorithm, stages, columns, params, functions,
                 times, accuracy_type, accuracies, data, alg_instance=None,
                 alg_params=None):
    if params.output_format == 'csv':
        output_csv(columns, params, functions, times, accuracies)
    elif params.output_format == 'json':
        output = []
        for i in range(len(stages)):
            result = gen_basic_dict(library, algorithm, stages[i], params,
                                    data[i], alg_instance, alg_params)
            result.update({'time[s]': times[i]})
            if accuracy_type is not None:
                result.update({f'{accuracy_type}': accuracies[i]})
            if hasattr(params, 'n_classes'):
                result['input_data'].update({'classes': params.n_classes})
            if hasattr(params, 'n_clusters'):
                if algorithm == 'kmeans':
                    result['input_data'].update(
                        {'n_clusters': params.n_clusters})
                elif algorithm == 'dbscan':
                    result.update({'n_clusters': params.n_clusters})
            # replace non-string init with string for kmeans benchmarks
            if alg_instance is not None:
                if 'init' in result['algorithm_parameters'].keys():
                    if not isinstance(result['algorithm_parameters']['init'], str):
                        result['algorithm_parameters']['init'] = 'random'
                if 'handle' in result['algorithm_parameters'].keys():
                    del result['algorithm_parameters']['handle']
            output.append(result)
        print(json.dumps(output, indent=4))


def import_fptype_getter():
    try:
        from daal4py.sklearn._utils import getFPType
    except:
        from daal4py.sklearn.utils import getFPType
    return getFPType
