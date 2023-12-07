# ===============================================================================
# Copyright 2020-2023 Intel Corporation
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

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

NumpyNumeric = Union[np.unsignedinteger, np.integer, np.floating]
Numeric = Union[int, float]
Scalar = Union[Numeric, bool, str, None]
JsonTypesUnion = Union[Scalar, List, Dict]
# TODO: replace Any with Union[Callable, ...]
ModuleContentMap = Dict[str, List[Any]]
# template may contain lists on first level
BenchTemplate = Dict[str, Any]
# case is expected to be nested dict
BenchCase = Dict[str, Dict[str, Any]]

Array = Union[pd.DataFrame, np.ndarray, csr_matrix]
