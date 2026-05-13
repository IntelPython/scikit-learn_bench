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


def compute_core_assignments(
    num_instances: int,
    cores_per_instance: int,
    numa_cpus_conf: Optional[Dict[int, str]] = None,
) -> List[str]:
    """
    Returns list of physcpubind strings for numactl, one per instance.

    NUMA-aware: keeps each instance within a single NUMA node when possible.
    Fallback: sequential blocks from core 0.
    Raises ValueError if insufficient cores.
    """
    total_needed = num_instances * cores_per_instance

    if numa_cpus_conf:
        numa_cores = {
            node: parse_cpu_range(cpu_str)
            for node, cpu_str in numa_cpus_conf.items()
        }
        total_available = sum(len(c) for c in numa_cores.values())
        if total_needed > total_available:
            raise ValueError(
                f"Need {total_needed} cores ({num_instances} instances x "
                f"{cores_per_instance} cores) but only {total_available} available"
            )

        assignments = []
        remaining = {node: list(cores) for node, cores in numa_cores.items()}

        for _ in range(num_instances):
            assigned = False
            for node in sorted(remaining.keys()):
                if len(remaining[node]) >= cores_per_instance:
                    instance_cores = remaining[node][:cores_per_instance]
                    remaining[node] = remaining[node][cores_per_instance:]
                    assignments.append(cores_to_range_str(instance_cores))
                    assigned = True
                    break
            if not assigned:
                # couldn't fit in a single node, take from multiple nodes
                instance_cores = []
                for node in sorted(remaining.keys()):
                    take = min(
                        len(remaining[node]),
                        cores_per_instance - len(instance_cores),
                    )
                    instance_cores.extend(remaining[node][:take])
                    remaining[node] = remaining[node][take:]
                    if len(instance_cores) == cores_per_instance:
                        break
                if len(instance_cores) < cores_per_instance:
                    raise ValueError("Insufficient cores for assignment")
                logger.warning(
                    f"Instance assigned cores across NUMA nodes: "
                    f"{cores_to_range_str(instance_cores)}"
                )
                assignments.append(cores_to_range_str(instance_cores))

        return assignments
    else:
        from psutil import cpu_count

        available = cpu_count(logical=True)
        if total_needed > available:
            raise ValueError(
                f"Need {total_needed} cores ({num_instances} instances x "
                f"{cores_per_instance} cores) but only {available} available"
            )
        assignments = []
        for i in range(num_instances):
            start = i * cores_per_instance
            end = start + cores_per_instance - 1
            assignments.append(f"{start}-{end}")
        return assignments
