from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.paths import scenario_cache_path

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]


CACHE_LAYOUT_VERSION = 3


def _ensure_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for benchmark caching.")


def cache_path(scenario_name: str) -> Path:
    return scenario_cache_path(scenario_name)


def has_cache(scenario_name: str) -> bool:
    return cache_path(scenario_name).is_file()


def save_tidy3d_result(
    scenario_name: str,
    *,
    frequencies: tuple[float, ...],
    monitors: dict[str, dict[str, Any]],
    cache_key: str | None = None,
) -> Path:
    _ensure_h5py()
    path = cache_path(scenario_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["layout_version"] = CACHE_LAYOUT_VERSION
        handle.attrs["scenario_name"] = scenario_name
        handle.attrs["frequencies_json"] = json.dumps([float(frequency) for frequency in frequencies])
        if cache_key is not None:
            handle.attrs["cache_key"] = str(cache_key)
        for monitor_name, monitor_data in monitors.items():
            group = handle.create_group(monitor_name)
            group.attrs["kind"] = str(monitor_data.get("kind", "field"))
            for key in ("axis", "normal_direction"):
                if key in monitor_data:
                    group.attrs[key] = str(monitor_data[key])
            if "position" in monitor_data:
                group.attrs["position"] = float(monitor_data["position"])
            if "center" in monitor_data:
                group.attrs["center"] = np.asarray(monitor_data["center"], dtype=np.float64)
            if "size" in monitor_data:
                group.attrs["size"] = np.asarray(monitor_data["size"], dtype=np.float64)
            if "frequencies" in monitor_data:
                group.create_dataset(
                    "frequencies",
                    data=np.asarray(monitor_data["frequencies"], dtype=np.float64),
                )
            for coord_key in ("x", "y", "z"):
                if coord_key in monitor_data:
                    group.create_dataset(coord_key, data=np.asarray(monitor_data[coord_key]))
            if "fields" in monitor_data:
                field_group = group.create_group("fields")
                for component, values in monitor_data["fields"].items():
                    array = np.asarray(values)
                    if np.iscomplexobj(array):
                        field_group.create_dataset(f"{component}_real", data=array.real)
                        field_group.create_dataset(f"{component}_imag", data=array.imag)
                    else:
                        field_group.create_dataset(component, data=array)
            for scalar_name in ("flux", "power"):
                if scalar_name in monitor_data:
                    group.create_dataset(scalar_name, data=np.asarray(monitor_data[scalar_name]))
    return path


def load_tidy3d_result(
    scenario_name: str,
    *,
    expected_cache_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    _ensure_h5py()
    path = cache_path(scenario_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark cache for {scenario_name}: {path}")

    out: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        layout_version = int(handle.attrs.get("layout_version", 0))
        if layout_version != CACHE_LAYOUT_VERSION:
            raise ValueError(
                f"Benchmark cache layout mismatch for {scenario_name}: "
                f"found {layout_version}, expected {CACHE_LAYOUT_VERSION}"
            )
        if expected_cache_key is not None:
            cache_key = handle.attrs.get("cache_key")
            if cache_key is None:
                raise ValueError(f"Benchmark cache for {scenario_name} is missing cache_key metadata.")
            actual_cache_key = cache_key if isinstance(cache_key, str) else cache_key.decode()
            if actual_cache_key != expected_cache_key:
                raise ValueError(
                    f"Benchmark cache key mismatch for {scenario_name}: "
                    f"found {actual_cache_key}, expected {expected_cache_key}"
                )
        for monitor_name in handle.keys():
            group = handle[monitor_name]
            monitor_data: dict[str, Any] = {"kind": group.attrs["kind"]}
            for key in ("axis", "normal_direction"):
                if key in group.attrs:
                    value = group.attrs[key]
                    monitor_data[key] = value if isinstance(value, str) else value.decode()
            if "position" in group.attrs:
                monitor_data["position"] = float(group.attrs["position"])
            if "center" in group.attrs:
                monitor_data["center"] = tuple(float(v) for v in group.attrs["center"])
            if "size" in group.attrs:
                monitor_data["size"] = tuple(float(v) for v in group.attrs["size"])
            if "frequencies" in group:
                monitor_data["frequencies"] = tuple(float(v) for v in group["frequencies"][:])
            for coord_key in ("x", "y", "z"):
                if coord_key in group:
                    monitor_data[coord_key] = group[coord_key][:]
            if "fields" in group:
                fields = {}
                field_group = group["fields"]
                component_names = sorted(
                    {name.replace("_real", "").replace("_imag", "") for name in field_group.keys()}
                )
                for component in component_names:
                    if f"{component}_real" in field_group:
                        fields[component] = field_group[f"{component}_real"][:] + 1j * field_group[f"{component}_imag"][:]
                    else:
                        fields[component] = field_group[component][:]
                monitor_data["fields"] = fields
            for scalar_name in ("flux", "power"):
                if scalar_name in group:
                    monitor_data[scalar_name] = group[scalar_name][:]
            out[monitor_name] = monitor_data
    return out
