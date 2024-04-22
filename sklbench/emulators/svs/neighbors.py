# ===============================================================================
# Copyright 2024 Intel Corporation
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

import pysvs
from psutil import cpu_count

from ..common.neighbors import NearestNeighborsBase


class NearestNeighbors(NearestNeighborsBase):
    """
    Minimal class emulating `sklearn.neighbors.NearestNeighbors` estimator
    """

    def __init__(
        self,
        n_neighbors=5,
        algorithm="vamana",
        metric="euclidean",
        graph_max_degree=64,
        window_size=128,
        n_jobs=cpu_count(logical=False),
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.graph_max_degree = graph_max_degree
        self.window_size = window_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        build_params = pysvs.VamanaBuildParameters(
            graph_max_degree=self.graph_max_degree,
            window_size=self.window_size,
            num_threads=self.n_jobs,
        )
        self._index = pysvs.Vamana.build(
            build_params,
            X,
            pysvs.DistanceType.L2,
            num_threads=self.n_jobs,
        )
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        indices, distances = self._index.search(X, k)
        if return_distance:
            return distances, indices
        else:
            return indices
