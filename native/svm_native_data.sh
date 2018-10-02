#!/bin/bash
# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT


for x in "data/two/X-"*.npy; do
    x_=`basename "$x"`
    y=`dirname "$x"`/y${x_:1}
    echo "Preparing data for native benchmark: $x $y"
    python "native/make_two_class_data.py" "$x" "$y" || exit 1
done
for x in "data/multi/X-"*.npy; do
    x_=`basename "$x"`
    y=`dirname "$x"`/y${x_:1}
    echo "Preparing data for native benchmark: $x $y"
    python "native/make_multi_class_data.py" "$x" "$y" || exit 1
done
