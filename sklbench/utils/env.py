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

import json
from typing import Dict

import pandas as pd

from .common import read_output_from_command
from .logger import logger


def get_numa_cpus_conf() -> Dict[int, str]:
    try:
        _, lscpu_text, _ = read_output_from_command("lscpu")
        return {
            i: numa_cpus
            for i, numa_cpus in enumerate(
                map(
                    lambda x: x.split(" ")[-1],
                    filter(
                        lambda line: "NUMA" in line and "CPU(s)" in line,
                        lscpu_text.split("\n"),
                    ),
                )
            )
        }
    except FileNotFoundError:
        logger.warning("Unable to get numa cpus configuration via lscpu")
        return dict()


def get_software_info() -> Dict:
    result = dict()
    # conda list
    try:
        _, conda_list, _ = read_output_from_command("conda list --json")
        conda_packages = json.loads(conda_list)
        result["conda_packages"] = {pkg.pop("name"): pkg for pkg in conda_packages}
    # pip list
    except (FileNotFoundError, PermissionError, AttributeError):
        logger.warning("Unable to get python packages list via conda")
        try:
            _, pip_list, _ = read_output_from_command("pip list --format json")
            pip_packages = json.loads(pip_list)
            result["pip_packages"] = {pkg.pop("name"): pkg for pkg in pip_packages}
        except (FileNotFoundError, PermissionError, AttributeError):
            logger.warning("Unable to get python packages list via pip")
    return result


def get_oneapi_devices() -> pd.DataFrame:
    try:
        import dpctl

        devices = dpctl.get_devices()
        devices = {
            device.filter_string: {
                "name": device.name,
                "vendor": device.vendor,
                "type": str(device.device_type).split(".")[1],
                "driver version": device.driver_version,
                "memory size[GB]": device.global_mem_size / 2**30,
            }
            for device in devices
        }
        if len(devices) > 0:
            return pd.DataFrame(devices).T
        else:
            logger.warning("dpctl device table is empty")
    except (ImportError, ModuleNotFoundError):
        logger.warning("dpctl can not be imported")
    # 'type' is left for device type selection only
    return pd.DataFrame({"type": list()})


def get_higher_isa(cpu_flags: str) -> str:
    # TODO: add non-x86 sets
    ordered_sets = ["avx512", "avx2", "avx", "sse4_2", "ssse3", "sse2"]
    for isa in ordered_sets:
        if isa in cpu_flags:
            return isa
    return "unknown"


def get_hardware_info() -> Dict:
    result = dict()
    oneapi_devices = get_oneapi_devices()
    if len(oneapi_devices) > 0:
        logger.info(f"DPCTL listed devices:\n{oneapi_devices}\n")
    # CPU
    try:
        from cpuinfo import get_cpu_info

        cpu_info = get_cpu_info()
        # remap cpu info values to better understandable names
        fields_map = {
            "arch": "architecture",
            "brand_raw": "name",
            "flags": "flags",
            "count": "logical_cpus",
        }
        for key in list(cpu_info.keys()):
            value = cpu_info.pop(key)
            if key in fields_map.keys():
                cpu_info[fields_map[key]] = value
        # squash CPU flags
        cpu_info["flags"] = " ".join(cpu_info["flags"])
        result["CPU"] = cpu_info
        logger.info(f'CPU name: {cpu_info["name"]}')
        logger.info(
            "Highest supported ISA: " f'{get_higher_isa(cpu_info["flags"]).upper()}'
        )
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse CPU info with "cpuinfo" module')
    # GPUs
    result["GPU(s)"] = dict()
    try:
        oneapi_gpus = oneapi_devices[oneapi_devices["type"] == "gpu"]
        result["GPU(s)"].update(oneapi_gpus.T.to_dict())
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to get devices with "dpctl" module')
    # RAM size
    try:
        import psutil

        result["RAM size[GB]"] = psutil.virtual_memory().total / 2**30
        logger.info(f'RAM size[GB]: {round(result["RAM size[GB]"], 3)}')
    except (ImportError, ModuleNotFoundError):
        logger.warning('Unable to parse memory info with "psutil" module')
    return result


def get_environment_info() -> Dict:
    return {"hardware": get_hardware_info(), "software": get_software_info()}
