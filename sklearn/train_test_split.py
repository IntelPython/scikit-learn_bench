# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import measure_function_time, parse_args, load_data, print_output
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='scikit-learn train_test_split benchmark')
parser.add_argument('--train-size', type=float, default=0.75,
                    help='Size of training subset')
parser.add_argument('--test-size', type=float, default=0.25,
                    help='Size of testing subset')
parser.add_argument('--shuffle', default=False, action='store_true',
                    help='Perform data shuffle before splitting')
parser.add_argument('--exclude-y', default=False, action='store_true',
                    help='Exclude label (Y) in splitting')
parser.add_argument('--rng', default=None,
                    choices=('MT19937', 'SFMT19937', 'MT2203', 'R250', 'WH',
                             'MCG31', 'MCG59', 'MRG32K3A', 'PHILOX4X32X10',
                             'NONDETERM', None),
                    help='Random numbers generator for shuffling '
                         '(only for IDP scikit-learn)')
params = parse_args(parser)

# Load generated data
X, y, _, _ = load_data(params)

if params.exclude_y:
    data_args = (X, )
else:
    data_args = (X, y)

tts_params = {
    'train_size': params.train_size,
    'test_size': params.test_size,
    'shuffle': params.shuffle,
    'random_state': params.seed
}

if params.rng is not None:
    tts_params['rng'] = params.rng

time, _ = measure_function_time(train_test_split, *data_args, **tts_params,
                                params=params)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

print_output(library='sklearn', algorithm='train_test_split',
             stages=['training'], columns=columns, params=params,
             functions=['train_test_split'], times=[time], accuracies=[None],
             accuracy_type=None, data=[X], alg_params=tts_params)
