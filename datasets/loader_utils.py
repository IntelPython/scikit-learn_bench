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
import requests
import os
from urllib.request import urlretrieve
from shutil import copyfile
import numpy as np
from tqdm import tqdm


def retrieve(url: str, filename: str) -> None:
    # rewritting urlretrieve without using urllib library,
    # otherwise it would fail codefactor test due to security issues.
    if os.path.isfile(url):
        # reporthook is ignored for local urls
        copyfile(url, filename)
    elif url.startswith('http'):
        response = requests.get(url,stream=True)
        if response.status_code != 200:
            raise AssertionError(f"Failed to download from {url},\n"+\
                "Response returned status code {response.status_code}")
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        pbar = tqdm(total=total_size/1024, unit='kB')
        with open(filename, 'wb+') as file:
            for data in response.iter_content(block_size):
                pbar.update(len(data)/1024)
                file.write(data)
        pbar.close()
        if total_size != 0 and pbar.n != total_size/1024:
            raise AssertionError("Some content was present but not downloaded/written")


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
