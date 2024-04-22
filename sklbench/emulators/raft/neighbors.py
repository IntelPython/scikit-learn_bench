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

import cupy as cp
from pylibraft.common import DeviceResources
from pylibraft.neighbors import brute_force, cagra, ivf_flat, ivf_pq

from ..common import NearestNeighborsBase


class NearestNeighbors(NearestNeighborsBase):
    """
    Minimal class emulating `sklearn.neighbors.NearestNeighbors` estimator
    """

    def __init__(
        self,
        n_neighbors=5,
        algorithm="brute",
        metric="euclidean",
        n_lists=1024,
        n_probes=64,
        m_subvectors=16,
        n_bits=8,
        intermediate_graph_degree=128,
        graph_degree=64,
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.m_subvectors = m_subvectors
        self.n_bits = n_bits
        self.intermediate_graph_degree = intermediate_graph_degree
        self.graph_degree = graph_degree
        self._handle = DeviceResources()

    def fit(self, X, y=None):
        d = X.shape[1]
        if isinstance(self.m_subvectors, float):
            self.m_subvectors = self.get_m_subvectors(self.m_subvectors, d)
        if self.algorithm == "brute":
            self._X_fit = X
        elif self.algorithm == "ivf_flat":
            index_params = ivf_flat.IndexParams(n_lists=self.n_lists, metric=self.metric)
            self._index = ivf_flat.build(index_params, X, handle=self._handle)
        elif self.algorithm == "ivf_pq":
            index_params = ivf_pq.IndexParams(
                n_lists=self.n_lists,
                metric=self.metric,
                pq_dim=self.m_subvectors,
                pq_bits=self.n_bits,
            )
            self._index = ivf_pq.build(index_params, X, handle=self._handle)
        elif self.algorithm == "cagra":
            index_params = cagra.IndexParams(
                metric="sqeuclidean",
                intermediate_graph_degree=self.intermediate_graph_degree,
                graph_degree=self.graph_degree,
            )
            self._index = cagra.build(index_params, X, handle=self._handle)
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")
        self._handle.sync()
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if self.algorithm == "brute":
            distances, indices = brute_force.knn(
                self._X_fit, X, k, metric=self.metric, handle=self._handle
            )
        elif self.algorithm == "ivf_flat":
            distances, indices = ivf_flat.search(
                ivf_flat.SearchParams(n_probes=self.n_probes),
                self._index,
                X,
                k + 1,
                handle=self._handle,
            )
        elif self.algorithm == "ivf_pq":
            distances, indices = ivf_pq.search(
                ivf_pq.SearchParams(n_probes=self.n_probes),
                self._index,
                X,
                k,
                handle=self._handle,
            )
        elif self.algorithm == "cagra":
            distances, indices = cagra.search(
                cagra.SearchParams(itopk_size=int(2 * k)),
                self._index,
                X,
                k,
                handle=self._handle,
            )
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")
        self._handle.sync()
        if not isinstance(distances, cp.ndarray):
            distances = cp.asarray(distances)
        if not isinstance(indices, cp.ndarray):
            indices = cp.asarray(indices)
        if self.algorithm == "ivf_flat":
            distances, indices = distances[:, :-1], indices[:, :-1]
        if return_distance:
            return distances, indices
        else:
            return indices
