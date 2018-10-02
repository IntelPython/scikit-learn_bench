# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

def getArguments(argParser):
    argParser.add_argument('--iteration',   default='?',       help="iteration")
    argParser.add_argument('--core-number', default='?',       help="core number")
    argParser.add_argument('--arch',        default='?',       help="architecture")
    argParser.add_argument('--batchID',     default='?',       help="batchID")
    argParser.add_argument('--prefix',      default='?',       help="prefix string")
    argParser.add_argument('--place',       default='?',       help="prefix string")
    argParser.add_argument('--cache',       default='?',       help="cached/non-cached")
    argParser.add_argument('--size',        default='?',       help="array size", nargs='*')
    args = argParser.parse_args()

    args.size = [int(n) for n in args.size[0].split(',')]
    return args
