# ===============================================================================
# Copyright 2026 Intel Corporation
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

import os
from typing import Dict, List, Optional

from .logger import logger


def parse_cpu_range(range_str: str) -> List[int]:
    """Parse '0-11,24-35' into sorted list of individual core IDs."""
    cores = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            cores.extend(range(int(start), int(end) + 1))
        else:
            cores.append(int(part))
    return sorted(cores)


def cores_to_range_str(cores: List[int]) -> str:
    """Convert [0,1,2,3,8,9,10,11] to '0-3,8-11'."""
    if not cores:
        return ""
    cores = sorted(cores)
    ranges = []
    start = cores[0]
    end = cores[0]
    for c in cores[1:]:
        if c == end + 1:
            end = c
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = c
            end = c
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def is_consecutive(cores: List[int]) -> bool:
    """Check if a sorted list of core IDs is consecutive."""
    for i in range(1, len(cores)):
        if cores[i] != cores[i - 1] + 1:
            return False
    return True


def get_numa_node_for_core(core_id: int, numa_cpus_conf: Dict[int, str]) -> int:
    """Return NUMA node ID for a given core, or -1 if unknown."""
    for node, cpu_str in numa_cpus_conf.items():
        if core_id in parse_cpu_range(cpu_str):
            return node
    return -1


def compute_core_assignments(
    num_instances: int,
    cores_per_instance: int,
    numa_cpus_conf: Optional[Dict[int, str]] = None,
) -> List[str]:
    """
    Returns list of physcpubind strings for numactl, one per instance.

    Uses only CPUs available to the current process (respects parent's taskset).
    Splits available cores into sequential groups of cores_per_instance.
    Warns if a group is non-consecutive or spans NUMA nodes.
    Raises ValueError if the process doesn't have enough cores.
    """
    available_cores = sorted(os.sched_getaffinity(0))
    total_needed = num_instances * cores_per_instance

    if total_needed > len(available_cores):
        raise ValueError(
            f"Need {total_needed} cores ({num_instances} instances x "
            f"{cores_per_instance} cores) but only {len(available_cores)} "
            f"available to this process"
        )

    # Build NUMA lookup if available
    numa_lookup = {}
    if numa_cpus_conf:
        for node, cpu_str in numa_cpus_conf.items():
            for core in parse_cpu_range(cpu_str):
                numa_lookup[core] = node

    assignments = []
    for i in range(num_instances):
        instance_cores = available_cores[
            i * cores_per_instance : (i + 1) * cores_per_instance
        ]

        if not is_consecutive(instance_cores):
            logger.warning(
                f"Instance {i}: assigned non-consecutive cores "
                f"{cores_to_range_str(instance_cores)}"
            )

        if numa_lookup:
            nodes = set(numa_lookup.get(c, -1) for c in instance_cores)
            if len(nodes) > 1:
                logger.warning(
                    f"Instance {i}: cores {cores_to_range_str(instance_cores)} "
                    f"span multiple NUMA nodes {sorted(nodes)}"
                )

        assignments.append(cores_to_range_str(instance_cores))

    return assignments
