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
import sys

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.utils import check_random_state


def gen_blobs(args):
    X, y = make_blobs(n_samples=args.samples + args.test_samples,
                      n_features=args.features,
                      centers=args.clusters,
                      center_box=(-32, 32),
                      shuffle=True,
                      random_state=args.seed)
    np.save(args.filex, X[:args.samples])
    if args.test_samples != 0:
        np.save(args.filextest, X[args.samples:])
    return 0


def gen_regression(args):
    rs = check_random_state(args.seed)
    X, y = make_regression(n_targets=1,
                           n_samples=args.samples + args.test_samples,
                           n_features=args.features,
                           n_informative=args.features,
                           bias=rs.normal(0, 3),
                           random_state=rs)
    np.save(args.filex, X[:args.samples])
    np.save(args.filey, y[:args.samples])
    if args.test_samples != 0:
        np.save(args.filextest, X[args.samples:])
        np.save(args.fileytest, y[args.samples:])
    return 0


def gen_classification(args):
    X, y = make_classification(n_samples=args.samples + args.test_samples,
                               n_features=args.features,
                               n_informative=args.features,
                               n_repeated=0,
                               n_redundant=0,
                               n_classes=args.classes,
                               random_state=args.seed)
    np.save(args.filex, X[:args.samples])
    np.save(args.filey, y[:args.samples])
    if args.test_samples != 0:
        np.save(args.filextest, X[args.samples:])
        np.save(args.fileytest, y[args.samples:])
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Dataset generator using scikit-learn')
    parser.add_argument('-f', '--features', type=int, default=1000,
                        help='Number of features in dataset')
    parser.add_argument('-s', '--samples', type=int, default=10000,
                        help='Number of samples in dataset')
    parser.add_argument('--ts', '--test-samples', type=int, default=0,
                        dest='test_samples',
                        help='Number of test samples in dataset')
    parser.add_argument('-d', '--seed', type=int, default=0,
                        help='Seed for random state')
    subparsers = parser.add_subparsers(dest='problem')
    subparsers.required = True

    regr_parser = subparsers.add_parser('regression',
                                        help='Regression data')
    regr_parser.set_defaults(func=gen_regression)
    regr_parser.add_argument('-x', '--filex', '--fileX', type=str,
                             required=True, help='Path to save matrix X')
    regr_parser.add_argument('-y', '--filey', '--fileY', type=str,
                             required=True, help='Path to save vector y')
    regr_parser.add_argument('--xt', '--filextest', '--fileXtest', type=str,
                             dest='filextest',
                             help='Path to save test matrix X')
    regr_parser.add_argument('--yt', '--fileytest', '--fileYtest', type=str,
                             dest='fileytest',
                             help='Path to save test vector y')

    clsf_parser = subparsers.add_parser('classification',
                                        help='Classification data')
    clsf_parser.set_defaults(func=gen_classification)
    clsf_parser.add_argument('-c', '--classes', type=int, default=5,
                             help='Number of classes')
    clsf_parser.add_argument('-x', '--filex', '--fileX', type=str,
                             required=True, help='Path to save matrix X')
    clsf_parser.add_argument('-y', '--filey', '--fileY', type=str,
                             required=True,
                             help='Path to save label vector y')
    clsf_parser.add_argument('--xt', '--filextest', '--fileXtest', type=str,
                             dest='filextest',
                             help='Path to save test matrix X')
    clsf_parser.add_argument('--yt', '--fileytest', '--fileYtest', type=str,
                             dest='fileytest',
                             help='Path to save test vector y')

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
