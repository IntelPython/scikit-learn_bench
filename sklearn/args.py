# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


def getArguments(argParser):
    argParser.add_argument('--iteration', default=10, type=int,
                           help='Number of repetitions to run')
    argParser.add_argument('--num-threads', '--core-number', default=-1,
                           type=int, help='Number of threads to use')
    argParser.add_argument('--arch', default='?',
                           help='Machine architecture, for bookkeeping')
    argParser.add_argument('--batchID', default='?',
                           help='Batch ID, for bookkeeping')
    argParser.add_argument('--prefix', default='?',
                           help='Prefix string, for bookkeeping')
    argParser.add_argument('--place',       default='?',       help="prefix string")
    argParser.add_argument('--cache',       default='?',       help="cached/non-cached")
    argParser.add_argument('--size', default='?',
                           help="array size, delimited by comma or 'x'")
    args = argParser.parse_args()

    args.size = [int(n) for n in args.size.replace('x', ',').split(',')]
    return args


def coreString(num):
    return 'Serial' if num == 1 else 'Threaded'
