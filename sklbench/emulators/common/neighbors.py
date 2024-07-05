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


from warnings import warn

import numpy as np


class NearestNeighborsBase:
    def get_params(self):
        result = {
            "n_neighbors": self.n_neighbors,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "metric_params": None,
            "p": 2 if "euclidean" in self.metric else None,
        }
        optional_keys = [
            "n_lists",
            "n_probes",
            "m_subvectors",
            "n_bits",
            "intermediate_graph_degree",
            "graph_degree",
        ]
        for optional_key in optional_keys:
            if hasattr(self, optional_key):
                result[optional_key] = getattr(self, optional_key)
        return result

    def get_m_subvectors(self, percentile, d):
        """Method to get `m_subvectors` closest to specific percentile and
        compatible with RAFT and FAISS"""
        raft_comp = np.arange(1, d // 16) * 16
        faiss_comp = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48])
        faiss_comp = faiss_comp[d % faiss_comp == 0]
        intersection = np.intersect1d(raft_comp, faiss_comp)
        if len(intersection) == 0:
            m_subvectors = 16
            warn(
                f"Unable to calculate compatible m_subvectors from {d} features. "
                "Defaulting to 16 subvectors."
            )
        else:
            m_subvectors = int(
                intersection[np.argmin(np.abs(intersection - d * percentile))]
            )
        return m_subvectors
