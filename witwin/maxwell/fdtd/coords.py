from __future__ import annotations

import numpy as np

from ..monitors import normalize_component
from ..scene import prepare_scene


def _scene_axis_coords(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_scene = prepare_scene(scene)
    return (
        np.asarray(resolved_scene.x.detach().cpu().numpy(), dtype=np.float64),
        np.asarray(resolved_scene.y.detach().cpu().numpy(), dtype=np.float64),
        np.asarray(resolved_scene.z.detach().cpu().numpy(), dtype=np.float64),
    )


def component_coords(scene, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_scene = prepare_scene(scene)
    component_name = normalize_component(component)
    x, y, z = _scene_axis_coords(resolved_scene)

    if component_name == "ex":
        return x[:-1] + 0.5 * float(resolved_scene.dx), y, z
    if component_name == "ey":
        return x, y[:-1] + 0.5 * float(resolved_scene.dy), z
    if component_name == "ez":
        return x, y, z[:-1] + 0.5 * float(resolved_scene.dz)
    if component_name == "hx":
        return x, y[:-1] + 0.5 * float(resolved_scene.dy), z[:-1] + 0.5 * float(resolved_scene.dz)
    if component_name == "hy":
        return x[:-1] + 0.5 * float(resolved_scene.dx), y, z[:-1] + 0.5 * float(resolved_scene.dz)
    if component_name == "hz":
        return x[:-1] + 0.5 * float(resolved_scene.dx), y[:-1] + 0.5 * float(resolved_scene.dy), z
    raise ValueError(f"Unsupported field component: {component!r}")


def centered_cell_coords(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_scene = prepare_scene(scene)
    x, y, z = _scene_axis_coords(resolved_scene)
    return (
        x[:-1] + 0.5 * float(resolved_scene.dx),
        y[:-1] + 0.5 * float(resolved_scene.dy),
        z[:-1] + 0.5 * float(resolved_scene.dz),
    )
