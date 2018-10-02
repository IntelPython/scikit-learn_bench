# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import sys
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default='data/', help='destination dir')
    parser.add_argument('--vectors', '-v', default=10000, type=int,
                        help='number of vectors in dataset')
    parser.add_argument('--features', '-f', default=1000, type=int,
                        help='number of features in vector')
    args = parser.parse_args()

    features = args.features
    vectors = args.vectors

    # for two-class problem
    from make_svm_datasets import gen_datasets as get_datasets
    two_path = os.path.join(args.dest, 'two')
    os.makedirs(two_path, exist_ok=True)
    get_datasets([features], [vectors], 2, dest=two_path)

    # for multi-class problem
    multi_path = os.path.join(args.dest, 'multi')
    os.makedirs(multi_path, exist_ok=True)
    get_datasets([features], [vectors], 5, dest=multi_path)

if __name__ == '__main__':
    main()
