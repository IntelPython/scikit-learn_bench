# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import measure_function_time, parse_args, load_data, print_output
from cuml import train_test_split

parser = argparse.ArgumentParser(
    description='cuml train_test_split benchmark')
parser.add_argument('--train-size', type=float, default=0.75,
                    help='Size of training subset')
parser.add_argument('--test-size', type=float, default=0.25,
                    help='Size of testing subset')
parser.add_argument('--do-not-shuffle', default=False, action='store_true',
                    help='Do not perform data shuffle before splitting')
params = parse_args(parser)

# Load generated data
X, y, _, _ = load_data(params)

tts_params = {
    'train_size': params.train_size,
    'test_size': params.test_size,
    'shuffle': not params.do_not_shuffle,
    'random_state': params.seed
}

time, _ = measure_function_time(train_test_split, X=X, y=y, params=params)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

print_output(library='cuml', algorithm='train_test_split',
             stages=['training'], columns=columns, params=params,
             functions=['train_test_split'], times=[time], accuracies=[None],
             accuracy_type=None, data=[X], alg_params=tts_params)
