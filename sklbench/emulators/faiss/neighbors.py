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


import faiss

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
        device="cpu",
    ):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.m_subvectors = m_subvectors
        self.n_bits = n_bits
        self.device = device
        if self.device == "gpu":
            self._gpu_resources = faiss.StandardGpuResources()

    def fit(self, X, y=None):
        d = X.shape[1]
        if isinstance(self.m_subvectors, float):
            self.m_subvectors = self.get_m_subvectors(self.m_subvectors, d)
        self._base_index = faiss.IndexFlatL2(d)
        if self.algorithm == "brute":
            self._index = self._base_index
        elif self.algorithm == "ivf_flat":
            self._index = faiss.IndexIVFFlat(
                self._base_index, d, self.n_lists, faiss.METRIC_L2
            )
        elif self.algorithm == "ivf_pq":
            self._index = faiss.IndexIVFPQ(
                self._base_index,
                d,
                self.n_lists,
                self.m_subvectors,
                self.n_bits,
                faiss.METRIC_L2,
            )
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")
        if self.device == "gpu":
            self._index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self._index)
        self._index.nprobe = self.n_probes
        self._index.train(X)
        self._index.add(X)
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        distances, indices = self._index.search(X, k)
        if return_distance:
            return distances, indices
        else:
            return indices
