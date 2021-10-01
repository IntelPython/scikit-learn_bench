# ===============================================================================
# Copyright 2020-2021 Intel Corporation
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
# ===============================================================================

import re
from urllib.request import urlretrieve, Request
import os
import numpy as np
import tqdm

pbar: tqdm.tqdm = None

def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit='kB')

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None


def retrieve(url: str, filename: str) -> None:
    if url.lower().startswith('http'):
        req = Request(url)
    elif not os.path.isfile(url):
        raise ValueError, None
    urlretrieve(url, filename, reporthook=_show_progress) #nosec


def read_libsvm_msrank(file_obj, n_samples, n_features, dtype):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples,))

    counter = 0

    regexp = re.compile(r'[A-Za-z0-9]+:(-?\d*\.?\d+)')

    for line in file_obj:
        line = str(line).replace("\\n'", "")
        line = regexp.sub(r'\g<1>', line)
        line = line.rstrip(" \n\r").split(' ')

        y[counter] = int(line[0])
        X[counter] = [float(i) for i in line[1:]]

        counter += 1
        if counter == n_samples:
            break

    return np.array(X, dtype=dtype), np.array(y, dtype=dtype)


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def count_lines(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)
