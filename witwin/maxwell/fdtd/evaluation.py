import itertools

import numpy as np
import torch

import witwin.maxwell as mw
from .solver import calculate_required_steps


def build_empty_scene(
    *,
    domain_half,
    grid,
    device,
    pml_thickness,
    pml_strength,
    source_width=0.05,
    source_amplitude=100.0,
    polarization=(0.0, 0.0, 1.0),
    source_frequency=1e9,
):
    resolution = (2.0 * domain_half) / grid
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-domain_half, domain_half),
                (-domain_half, domain_half),
                (-domain_half, domain_half),
            )
        ),
        grid=mw.GridSpec.uniform(resolution),
        boundary=mw.BoundarySpec.pml(num_layers=pml_thickness, strength=pml_strength),
        device=device,
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization=polarization,
            width=source_width,
            source_time=mw.CW(frequency=source_frequency, amplitude=source_amplitude),
            name="src",
        )
    )
    return scene


def compute_steps(scene, dt, c, frequency, num_cycles=20, transient_cycles=20):
    bounds = scene.domain.bounds
    domain_size = max(
        bounds[0][1] - bounds[0][0],
        bounds[1][1] - bounds[1][0],
        bounds[2][1] - bounds[2][0],
    )
    return calculate_required_steps(
        frequency=frequency,
        dt=dt,
        c=c,
        num_cycles=num_cycles,
        transient_cycles=transient_cycles,
        domain_size=domain_size,
    )


def run_empty_scene_solver(
    *,
    domain_half,
    grid,
    frequency,
    absorber_type,
    cpml_config,
    pml_thickness,
    pml_strength,
    device,
    source_width=0.05,
    source_amplitude=100.0,
    num_cycles=20,
    transient_cycles=20,
):
    scene = build_empty_scene(
        domain_half=domain_half,
        grid=grid,
        device=device,
        pml_thickness=pml_thickness,
        pml_strength=pml_strength,
        source_width=source_width,
        source_amplitude=source_amplitude,
        source_frequency=frequency,
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequency=frequency,
        absorber=absorber_type,
        cpml_config=cpml_config,
        run_time=mw.TimeConfig.auto(steady_cycles=num_cycles, transient_cycles=transient_cycles),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
    )
    prepared = sim.prepare()
    steps = compute_steps(
        scene,
        prepared.solver.dt,
        prepared.solver.c,
        frequency,
        num_cycles=num_cycles,
        transient_cycles=transient_cycles,
    )
    prepared.simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = prepared.run()
    field_mag, x_coords, y_coords, z_coords = result.solver._interpolate_yee_to_center(result.raw_output)
    return {
        "field": field_mag,
        "x": x_coords,
        "y": y_coords,
        "z": z_coords,
        "steps": steps,
    }


def crop_reference_field(reference_field, reference_coords, target_coords):
    start = int(np.argmin(np.abs(reference_coords - target_coords[0])))
    end = start + target_coords.shape[0]
    return reference_field[start:end]


def extract_aligned_reference(reference, target):
    ref_field = reference["field"]
    x_start = int(np.argmin(np.abs(reference["x"] - target["x"][0])))
    y_start = int(np.argmin(np.abs(reference["y"] - target["y"][0])))
    z_start = int(np.argmin(np.abs(reference["z"] - target["z"][0])))
    x_end = x_start + target["field"].shape[0]
    y_end = y_start + target["field"].shape[1]
    z_end = z_start + target["field"].shape[2]
    return ref_field[x_start:x_end, y_start:y_end, z_start:z_end]


def evaluate_reference_error(
    target,
    reference,
    *,
    pml_thickness,
    source_exclusion_cells=2,
):
    ref_crop = extract_aligned_reference(reference, target)
    field = target["field"]
    abs_diff = np.abs(field - ref_crop)

    max_margin = max(min(field.shape) // 2 - 2, 0)
    effective_margin = min(int(pml_thickness), max_margin)
    if effective_margin == 0:
        interior = (slice(None),) * 3
    else:
        interior = (slice(effective_margin, -effective_margin),) * 3
    field_interior = field[interior]
    ref_interior = ref_crop[interior]
    diff_interior = abs_diff[interior]

    if field_interior.size == 0:
        field_interior = field
        ref_interior = ref_crop
        diff_interior = abs_diff

    center = np.array(field.shape) // 2
    grid = np.indices(field_interior.shape).transpose(1, 2, 3, 0)
    shifted = grid - (np.array(field_interior.shape) // 2)
    radius_sq = np.sum(shifted**2, axis=-1)
    exclusion_radius = min(source_exclusion_cells, max(min(field_interior.shape) // 4, 0))
    keep_mask = radius_sq >= exclusion_radius**2

    field_masked = field_interior[keep_mask]
    ref_masked = ref_interior[keep_mask]
    diff_masked = diff_interior[keep_mask]
    if diff_masked.size == 0:
        field_masked = field_interior.reshape(-1)
        ref_masked = ref_interior.reshape(-1)
        diff_masked = diff_interior.reshape(-1)

    rel_l2 = float(np.linalg.norm(diff_masked) / max(np.linalg.norm(ref_masked), 1e-12))
    rel_linf = float(np.max(diff_masked) / max(np.max(np.abs(ref_masked)), 1e-12))

    cx, cy, cz = center
    if effective_margin == 0:
        line_slice = slice(None)
    else:
        line_slice = slice(effective_margin, field.shape[0] - effective_margin)
    axis_target = field[line_slice, cy, cz]
    axis_ref = ref_crop[line_slice, cy, cz]
    axis_rel_l2 = float(np.linalg.norm(axis_target - axis_ref) / max(np.linalg.norm(axis_ref), 1e-12))

    shell_start = min(effective_margin, max(field.shape[0] - 2, 0))
    shell_end = min(shell_start + 2, field.shape[0])
    shell = field[shell_start:shell_end, :, :]
    shell_mean = float(np.mean(shell)) if shell.size > 0 else 0.0
    core_half = max(field.shape[0] // 4, 2)
    core = field[
        max(cx - core_half // 2, 0) : min(cx + core_half // 2, field.shape[0]),
        max(cy - core_half // 2, 0) : min(cy + core_half // 2, field.shape[1]),
        max(cz - core_half // 2, 0) : min(cz + core_half // 2, field.shape[2]),
    ]
    core_mean = float(np.mean(core))
    shell_to_core = shell_mean / max(core_mean, 1e-12)

    return {
        "reference_rel_l2": rel_l2,
        "reference_rel_linf": rel_linf,
        "axis_rel_l2": axis_rel_l2,
        "shell_to_core_ratio": shell_to_core,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "reference_field": ref_crop,
        "abs_diff_field": abs_diff,
    }


def generate_candidate_configs():
    grading_orders = [3.0, 4.0]
    kappa_max_values = [5.0, 8.0, 11.0]
    alpha_max_values = [0.02, 0.05, 0.10]
    reflections = [1e-6, 1e-8]
    for grading_order, kappa_max, alpha_max, reflection in itertools.product(
        grading_orders,
        kappa_max_values,
        alpha_max_values,
        reflections,
    ):
        yield {
            "grading_order": grading_order,
            "kappa_max": kappa_max,
            "alpha_max": alpha_max,
            "reflection": reflection,
        }


def tune_cpml_against_reference(
    *,
    grid,
    domain_half,
    reference_scale,
    frequency,
    pml_thickness,
    pml_strength,
    device,
    candidate_configs=None,
    num_cycles=20,
    transient_cycles=20,
):
    if candidate_configs is None:
        candidate_configs = list(generate_candidate_configs())

    reference = run_empty_scene_solver(
        domain_half=domain_half * reference_scale,
        grid=int(round(grid * reference_scale)),
        frequency=frequency,
        absorber_type="cpml",
        cpml_config=None,
        pml_thickness=pml_thickness,
        pml_strength=pml_strength,
        device=device,
        num_cycles=num_cycles,
        transient_cycles=transient_cycles,
    )

    results = []
    for config in candidate_configs:
        target = run_empty_scene_solver(
            domain_half=domain_half,
            grid=grid,
            frequency=frequency,
            absorber_type="cpml",
            cpml_config=config,
            pml_thickness=pml_thickness,
            pml_strength=pml_strength,
            device=device,
            num_cycles=num_cycles,
            transient_cycles=transient_cycles,
        )
        metrics = evaluate_reference_error(target, reference, pml_thickness=pml_thickness)
        score = metrics["reference_rel_l2"] + 0.5 * metrics["axis_rel_l2"] + 0.1 * metrics["shell_to_core_ratio"]
        results.append(
            {
                "config": dict(config),
                "score": float(score),
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
            }
        )

    results.sort(key=lambda item: item["score"])
    return {
        "reference": reference,
        "results": results,
        "best": results[0],
    }
