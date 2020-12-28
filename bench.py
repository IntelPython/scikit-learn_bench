#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import argparse
import numpy as np
import sklearn
import timeit
import json
import sys


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


def get_optimal_cache_size(n_rows, dtype=np.double, max_cache=64):
    '''
    Get an optimal cache size for sklearn.svm.SVC.

    Parameters
    ----------
    n_rows : int
        Number of rows in the dataset
    dtype : dtype-like, optional (default np.double)
        dtype to use for computing cache size
    max_cache : int, optional (default 64)
        Maximum cache size, in gigabytes
    '''

    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = byte_size * (n_rows ** 2)
    one_gb = 2 ** 30
    max_cache_bytes = max_cache * one_gb
    if optimal_cache_size_bytes > max_cache_bytes:
        return max_cache_bytes
    else:
        return optimal_cache_size_bytes


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
    parser.add_argument('--output-format', type=str, default='json',
                        choices=('json'), help='Output format: json')
    parser.add_argument('--time-method', type=str, default='box_filter',
                        choices=('box_filter'),
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
    parser.add_argument('--no-intel-optimized', default=False, action='store_true',
                        help='Use no intel optimized version. '
                             'Now avalible for scikit-learn benchmarks'),
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

    if not params.no_intel_optimized:
        try:
            from daal4py.sklearn import patch_sklearn
            patch_sklearn()
        except ImportError:
            print('Failed to import daal4py.sklearn.patch_sklearn.'
                  'Use stock version scikit-learn', file=sys.stderr)

    # disable finiteness check (default)
    if not params.check_finiteness:
        sklearn_disable_finiteness_check()

    # Ask DAAL what it thinks about this number of threads
    num_threads = prepare_daal_threads(num_threads=params.threads)
    if params.verbose:
        print(f'@ DAAL gave us {num_threads} threads')

    n_jobs = None
    if n_jobs_supported:
        n_jobs = num_threads = params.threads

    # Set threading and DAAL related params here
    setattr(params, 'threads', num_threads)
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


def set_daal_num_threads(num_threads):
    try:
        import daal4py
        if num_threads:
            daal4py.daalinit(nthreads=num_threads)
    except ImportError:
        print('@ Package "daal4py" was not found. Number of threads '
              'is being ignored')


def prepare_daal_threads(num_threads=-1):
    try:
        if num_threads > 0:
            set_daal_num_threads(num_threads)
        import daal4py
        num_threads = daal4py.num_threads()
    except ImportError:
        num_threads = 1

    return num_threads


def measure_function_time(func, *args, params, **kwargs):
    return time_box_filter(func, *args,
                           n_meas=params.box_filter_measurements,
                           time_limit=params.time_limit, **kwargs)


def time_box_filter(func, *args, n_meas, time_limit, **kwargs):
    times = []
    while len(times) < n_meas:
        t0 = timeit.default_timer()
        val = func(*args, **kwargs)
        t1 = timeit.default_timer()
        times.append(t1 - t0)
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
        if full_data[element] is not None and 'y' in element \
                and label_2d and hasattr(full_data[element], 'reshape'):
            full_data[element] = full_data[element].reshape(
                (full_data[element].shape[0], 1))
        # add dtype property to data if it's needed and doesn't exist
        if full_data[element] is not None and add_dtype and \
                not hasattr(full_data[element], 'dtype'):
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


def print_output(library, algorithm, stages, params, functions,
                 times, accuracy_type, accuracies, data, alg_instance=None,
                 alg_params=None):
    if params.output_format == 'json':
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
