import numpy as np
import torch

from ..visualization import (
    extract_orthogonal_slice,
    plot_cross_section_panels,
    plot_orthogonal_views,
)
from .postprocess import interpolate_yee_to_center


def plot_cross_section(
    solver,
    axis="z",
    position=0.0,
    component="abs",
    field_log_scale=False,
    figsize=(12, 5),
    save_path=None,
    verbose=None,
):
    if verbose is None:
        verbose = solver.verbose
    if solver.E_field is None:
        print("Run solve() first.")
        return

    ex, ey, ez = solver.E_field
    component_str = component.lower()

    if component_str == "abs":
        field_data, x_c, y_c, z_c = interpolate_yee_to_center(solver)
        eps_np = solver.scene.permittivity[:-1, :-1, :-1].cpu().numpy()
        title_prefix = "|E|"
    else:
        if component_str == "ex":
            field_data = torch.abs(ex).cpu().numpy()
        elif component_str == "ey":
            field_data = torch.abs(ey).cpu().numpy()
        elif component_str == "ez":
            field_data = torch.abs(ez).cpu().numpy()
        else:
            raise ValueError(f"Invalid component '{component}'. Use 'abs', 'Ex', 'Ey', or 'Ez'")
        x_c = solver.scene.x.cpu().numpy()
        y_c = solver.scene.y.cpu().numpy()
        z_c = solver.scene.z.cpu().numpy()
        eps_np = solver.scene.permittivity.cpu().numpy()
        title_prefix = f"|{component.upper()}|"

    field_info = extract_orthogonal_slice(field_data, axis, position, x_c, y_c, z_c)
    eps_info = extract_orthogonal_slice(eps_np, axis, position, x_c, y_c, z_c)

    field_slice = field_info["slice"]
    if field_log_scale:
        field_slice = np.where(field_slice > 1e-12, field_slice, 1e-12)
        fmax = np.max(field_slice)
        if fmax > 0:
            field_slice = 20 * np.log10(field_slice / fmax)
        cbar_label = f"{title_prefix} (dB)"
    else:
        cbar_label = f"{title_prefix} (V/m)"

    plot_cross_section_panels(
        field_slice=field_slice,
        permittivity_slice=eps_info["slice"],
        x_coords=field_info["x_coords"],
        y_coords=field_info["y_coords"],
        xlabel=field_info["xlabel"],
        ylabel=field_info["ylabel"],
        field_title=f"{title_prefix} at {axis}={position:.3f}m",
        field_cbar_label=cbar_label,
        permittivity_title=f"Permittivity at {axis}={position:.3f}m",
        figsize=figsize,
        save_path=save_path,
        verbose=verbose,
    )


def plot_isotropic_3views(
    solver,
    position=0.0,
    field_log_scale=True,
    figsize=(18, 5),
    save_path=None,
    vmin_db=-60,
    verbose=None,
):
    if verbose is None:
        verbose = solver.verbose
    if solver.E_field is None:
        print("Run solve() first.")
        return

    field_data, x_c, y_c, z_c = interpolate_yee_to_center(solver)
    eps = solver.scene.permittivity[:-1, :-1, :-1].cpu().numpy()

    plot_orthogonal_views(
        field_data=field_data,
        permittivity=eps,
        x_coords=x_c,
        y_coords=y_c,
        z_coords=z_c,
        position=position,
        field_log_scale=field_log_scale,
        figsize=figsize,
        save_path=save_path,
        vmin_db=vmin_db,
        verbose=verbose,
        suptitle="Total Electric Field Intensity |E| (FDFD)",
    )
