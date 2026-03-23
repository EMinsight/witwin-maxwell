from __future__ import annotations

from ..monitors import (
    FinitePlaneMonitor,
    FluxMonitor,
    ModeMonitor,
    PlaneMonitor,
    PointMonitor,
    normalize_component,
    normalize_axis,
)


def _point_observer_record(name, position, *, component: str) -> dict[str, object]:
    return {
        "name": str(name),
        "kind": "point",
        "position": tuple(float(value) for value in position),
        "component": normalize_component(component),
    }


def _plane_observer_record(name, axis, position, *, component: str) -> dict[str, object]:
    return {
        "name": str(name),
        "kind": "plane",
        "axis": normalize_axis(axis),
        "position": float(position),
        "component": normalize_component(component),
    }


def _internal_observer_name(monitor_name: str, component: str) -> str:
    return f"{monitor_name}::{normalize_component(component)}"


def compile_fdtd_observers(scene):
    compiled = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else scene.monitors
    for monitor in monitors:
        normalized_fields = tuple(normalize_component(field) for field in monitor.fields)
        if isinstance(monitor, PointMonitor):
            for field in normalized_fields:
                observer = _point_observer_record(
                    _internal_observer_name(monitor.name, field),
                    monitor.position,
                    component=field,
                )
                observer["monitor_name"] = monitor.name
                observer["monitor_fields"] = normalized_fields
                compiled.append(observer)
            continue
        if isinstance(monitor, (PlaneMonitor, FinitePlaneMonitor, FluxMonitor, ModeMonitor)):
            monitor_plane_position = monitor.position if isinstance(monitor, PlaneMonitor) else monitor.plane_position
            for field in normalized_fields:
                observer = _plane_observer_record(
                    _internal_observer_name(monitor.name, field),
                    axis=monitor.axis,
                    position=monitor_plane_position,
                    component=field,
                )
                observer["monitor_name"] = monitor.name
                observer["monitor_fields"] = normalized_fields
                observer["monitor_frequencies"] = monitor.frequencies
                observer["compute_flux"] = bool(getattr(monitor, "compute_flux", False))
                observer["normal_direction"] = getattr(monitor, "normal_direction", "+")
                if isinstance(monitor, ModeMonitor):
                    observer["monitor_type"] = "mode"
                    observer["mode_spec"] = monitor.mode_spec()
                compiled.append(observer)
            continue
        raise ValueError(f"Unsupported monitor type: {type(monitor).__name__}")
    return compiled
