"""
Unified visualization utilities for Maxwell solvers.

Sections:
  1. Core helpers (_to_numpy, slicing, etc.)
  2. 3D cross-section and orthogonal-view plotting
  3. FDTD time-domain animation (convergence viz)
  4. Differentiable FDFD gradient visualization
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# 1. Core helpers
# ============================================================================

def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _to_1d_numpy(coords):
    coords_np = _to_numpy(coords).astype(float)
    if coords_np.ndim != 1:
        raise ValueError("Coordinate arrays must be 1D")
    return coords_np


def nearest_index(coords, position):
    coords_np = _to_1d_numpy(coords)
    return int(np.argmin(np.abs(coords_np - position)))


def slice_extent(x_coords, y_coords, shape):
    x = _to_1d_numpy(x_coords)[: shape[0]]
    y = _to_1d_numpy(y_coords)[: shape[1]]

    def _axis_extent(values, count):
        if count == 0:
            return 0.0, 0.0
        if len(values) == 1:
            return float(values[0]), float(values[0])
        step = float(values[1] - values[0])
        return float(values[0]), float(values[0] + step * count)

    x0, x1 = _axis_extent(x, shape[0])
    y0, y1 = _axis_extent(y, shape[1])
    return [x0, x1, y0, y1]


# ============================================================================
# 2. 3D cross-section and orthogonal-view plotting
# ============================================================================

def extract_orthogonal_slice(data, axis, position, x_coords, y_coords, z_coords):
    axis = axis.lower()
    if axis == "x":
        idx = nearest_index(x_coords, position)
        slice_data = data[idx, :, :]
        plane_x = y_coords[: slice_data.shape[0]]
        plane_y = z_coords[: slice_data.shape[1]]
        xlabel, ylabel = "Y [m]", "Z [m]"
    elif axis == "y":
        idx = nearest_index(y_coords, position)
        slice_data = data[:, idx, :]
        plane_x = x_coords[: slice_data.shape[0]]
        plane_y = z_coords[: slice_data.shape[1]]
        xlabel, ylabel = "X [m]", "Z [m]"
    elif axis == "z":
        idx = nearest_index(z_coords, position)
        slice_data = data[:, :, idx]
        plane_x = x_coords[: slice_data.shape[0]]
        plane_y = y_coords[: slice_data.shape[1]]
        xlabel, ylabel = "X [m]", "Y [m]"
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return {
        "index": idx,
        "slice": slice_data,
        "x_coords": plane_x,
        "y_coords": plane_y,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "extent": slice_extent(plane_x, plane_y, slice_data.shape),
    }


def plot_slice_image(
    data, *, extent, xlabel, ylabel, title, colorbar_label,
    figsize=(10, 8), cmap="inferno", vmin=None, vmax=None,
    interpolation="bilinear", aspect="auto",
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    image = ax.imshow(
        _to_numpy(data).T, cmap=cmap, origin="lower", extent=extent,
        vmin=vmin, vmax=vmax, interpolation=interpolation, aspect=aspect,
    )
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_title(title, color="white", pad=20)
    ax.tick_params(colors="white")
    cbar = plt.colorbar(image, ax=ax, label=colorbar_label)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(colors="white")
    cbar.set_label(colorbar_label, color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    plt.grid(False)
    plt.tight_layout()
    return fig, ax


def plot_cross_section_panels(
    *, field_slice, permittivity_slice, x_coords, y_coords,
    xlabel, ylabel, field_title, field_cbar_label, permittivity_title,
    figsize=(12, 5), save_path=None, verbose=False,
    contour_levels=5, contour_color="white", field_vmin=None,
):
    field_np = _to_numpy(field_slice)
    eps_np = _to_numpy(permittivity_slice)
    extent = slice_extent(x_coords, y_coords, field_np.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("black")

    im1 = ax1.imshow(field_np.T, cmap="inferno", origin="lower", extent=extent,
                     interpolation="bilinear", vmin=field_vmin)
    ax1.set_xlabel(xlabel, color="white")
    ax1.set_ylabel(ylabel, color="white")
    ax1.set_title(field_title, color="white")
    ax1.tick_params(colors="white")
    ax1.set_facecolor("black")
    ax1.contour(
        _to_1d_numpy(x_coords)[: eps_np.shape[0]],
        _to_1d_numpy(y_coords)[: eps_np.shape[1]],
        eps_np.T, levels=contour_levels, colors=contour_color,
        linestyles="--", alpha=0.7,
    )
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label(field_cbar_label, color="white")
    cbar1.ax.tick_params(colors="white")

    im2 = ax2.imshow(eps_np.T, cmap="viridis", origin="lower", extent=extent,
                     interpolation="bilinear")
    ax2.set_xlabel(xlabel, color="white")
    ax2.set_ylabel(ylabel, color="white")
    ax2.set_title(permittivity_title, color="white")
    ax2.tick_params(colors="white")
    ax2.set_facecolor("black")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Relative permittivity", color="white")
    cbar2.ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor="black", edgecolor="none",
                    bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"Saved to: {save_path}")
        return None
    plt.show()
    return fig


def plot_orthogonal_views(
    *, field_data, permittivity, x_coords, y_coords, z_coords,
    position=0.0, field_log_scale=True, figsize=(18, 5), save_path=None,
    vmin_db=-60, verbose=False, suptitle="Total Electric Field Intensity |E|",
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("black")

    for ax, axis in zip(axes, ("x", "y", "z")):
        field_info = extract_orthogonal_slice(field_data, axis, position,
                                              x_coords, y_coords, z_coords)
        eps_info = extract_orthogonal_slice(permittivity, axis, position,
                                            x_coords, y_coords, z_coords)
        field_slice = _to_numpy(field_info["slice"])
        eps_slice = _to_numpy(eps_info["slice"])

        if field_log_scale:
            field_slice = np.where(field_slice > 1e-12, field_slice, 1e-12)
            field_max = np.max(field_slice)
            if field_max > 0:
                field_slice = 20 * np.log10(field_slice / field_max)
            cbar_label = "|E| (dB)"
            vmin = vmin_db
        else:
            cbar_label = "|E| (V/m)"
            vmin = None

        extent = field_info["extent"]
        image = ax.imshow(field_slice.T, cmap="inferno", origin="lower",
                          extent=extent, interpolation="bilinear", vmin=vmin)
        ax.contour(
            _to_1d_numpy(field_info["x_coords"])[: eps_slice.shape[0]],
            _to_1d_numpy(field_info["y_coords"])[: eps_slice.shape[1]],
            eps_slice.T, levels=[1.5], colors="cyan", linestyles="--", alpha=0.7,
        )
        ax.set_xlabel(field_info["xlabel"], color="white")
        ax.set_ylabel(field_info["ylabel"], color="white")
        ax.set_title(f"{axis.upper()}={position:.2f}m", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.set_facecolor("black")
        cbar = plt.colorbar(image, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, color="white")
        cbar.ax.tick_params(colors="white")

    plt.suptitle(suptitle, color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor="black", edgecolor="none",
                    bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"Saved to: {save_path}")
        return None
    plt.show()
    return fig


# ============================================================================
# 3. FDTD time-domain animation
# ============================================================================

def visualize_slice(solver, n, slice_type='xy', slice_index=None):
    """Visualize a field slice during FDTD time stepping."""
    if slice_index is None:
        slice_index = solver.Nz // 2 if slice_type == 'xy' else solver.Ny // 2

    Ex_c = 0.5 * (solver.Ex[:, :-1, :-1] + solver.Ex[:, 1:, :-1])
    Ey_c = 0.5 * (solver.Ey[:-1, :, :-1] + solver.Ey[1:, :, :-1])
    Ez_c = 0.5 * (solver.Ez[:-1, :-1, :] + solver.Ez[1:, :-1, :])
    E_magnitude = torch.sqrt(Ex_c**2 + Ey_c**2 + Ez_c**2)

    slices = {
        'xy': (E_magnitude[:, :, slice_index],
               (solver.scene.domain_range[0], solver.scene.domain_range[1]),
               (solver.scene.domain_range[2], solver.scene.domain_range[3]),
               'X [m]', 'Y [m]'),
        'xz': (E_magnitude[:, slice_index, :],
               (solver.scene.domain_range[0], solver.scene.domain_range[1]),
               (solver.scene.domain_range[4], solver.scene.domain_range[5]),
               'X [m]', 'Z [m]'),
        'yz': (E_magnitude[slice_index, :, :],
               (solver.scene.domain_range[2], solver.scene.domain_range[3]),
               (solver.scene.domain_range[4], solver.scene.domain_range[5]),
               'Y [m]', 'Z [m]'),
    }
    field_tensor, x_range, y_range, xlabel, ylabel = slices[slice_type]
    field_slice = field_tensor.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.imshow(field_slice.T, cmap='inferno', origin='lower',
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
               vmin=0, vmax=0.5, aspect='auto', interpolation='bilinear')
    plt.xlabel(xlabel, color='white')
    plt.ylabel(ylabel, color='white')
    t_ns = n * solver.dt * 1e9
    plt.title(f'3D FDTD - {slice_type.upper()} slice, t = {t_ns:.2f} ns',
              color='white', pad=20)
    plt.tick_params(colors='white', which='both')
    cbar = plt.colorbar(label='|E| field magnitude')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label('|E| field magnitude', color='white')
    cbar.ax.tick_params(colors='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.grid(False)
    plt.tight_layout()
    display(plt.gcf())
    clear_output(wait=True)
    plt.close()
    time.sleep(0.01)


def visualize_material_slice(solver, component='epsilon', slice_type='xy', slice_index=None):
    """Visualize material distribution (epsilon or mu)."""
    assert component in ['epsilon', 'mu']
    field = solver.epsilon_r if component == 'epsilon' else solver.mu_r

    if slice_index is None:
        slice_index = solver.Nz // 2 if slice_type == 'xy' else solver.Ny // 2

    slices = {
        'xy': (field[:, :, slice_index],
               (solver.scene.domain_range[0], solver.scene.domain_range[1]),
               (solver.scene.domain_range[2], solver.scene.domain_range[3]),
               'X [m]', 'Y [m]'),
        'xz': (field[:, slice_index, :],
               (solver.scene.domain_range[0], solver.scene.domain_range[1]),
               (solver.scene.domain_range[4], solver.scene.domain_range[5]),
               'X [m]', 'Z [m]'),
        'yz': (field[slice_index, :, :],
               (solver.scene.domain_range[2], solver.scene.domain_range[3]),
               (solver.scene.domain_range[4], solver.scene.domain_range[5]),
               'Y [m]', 'Z [m]'),
    }
    if slice_type not in slices:
        raise ValueError("slice_type must be 'xy', 'xz', or 'yz'")
    field_tensor, x_range, y_range, xlabel, ylabel = slices[slice_type]
    field_slice = field_tensor.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.imshow(field_slice.T, cmap='viridis', origin='lower',
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto')
    plt.xlabel(xlabel, color='white')
    plt.ylabel(ylabel, color='white')
    plt.title(f'{component}r distribution ({slice_type.upper()} slice)', color='white', pad=20)
    plt.tick_params(colors='white')
    cbar = plt.colorbar(label=f'{component}r value')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.set_label(f'{component}r', color='white')
    cbar.ax.tick_params(colors='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Differentiable FDFD gradient visualization
# ============================================================================

def plot_epsilon_gradient_field(circle_center_x=0.0, circle_center_y=0.0, radius=0.2,
                                 permittivity_value=4.0, grid_spacing=0.01,
                                 domain_range=(-1.0, 1.0, -1.0, 1.0), device='cpu',
                                 save_name='epsilon_gradient_field.png'):
    x_start, x_end, y_start, y_end = domain_range
    x = torch.linspace(x_start, x_end, int((x_end - x_start) / grid_spacing), device=device)
    y = torch.linspace(y_start, y_end, int((y_end - y_start) / grid_spacing), device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    Nx, Ny = len(x), len(y)
    aa_sharpness = 1.0 / grid_spacing

    theta = torch.tensor(circle_center_x, dtype=torch.float32, device=device, requires_grad=True)
    cy = torch.tensor(circle_center_y, dtype=torch.float32, device=device)
    r = torch.tensor(radius, dtype=torch.float32, device=device)

    dist = torch.sqrt((xx - theta)**2 + (yy - cy)**2)
    sdf = dist - r
    smooth_mask = torch.sigmoid(-sdf * aa_sharpness)
    epsilon = 1.0 * (1 - smooth_mask) + permittivity_value * smooth_mask

    with torch.no_grad():
        sigmoid_val = smooth_mask
        d_sigmoid_d_sdf = sigmoid_val * (1 - sigmoid_val) * (-aa_sharpness)
        d_dist_d_theta = (theta - xx) / (dist + 1e-10)
        grad_field = (permittivity_value - 1.0) * d_sigmoid_d_sdf * d_dist_d_theta

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(epsilon.detach().cpu().numpy().T, origin='lower',
                         extent=domain_range, cmap='viridis')
    axes[0].set_title(f'Epsilon Field (circle at theta={circle_center_x})')
    axes[0].set_xlabel('X [m]'); axes[0].set_ylabel('Y [m]')
    plt.colorbar(im1, ax=axes[0], label='epsilon')
    axes[0].add_patch(plt.Circle((circle_center_x, circle_center_y), radius,
                                  fill=False, color='white', linestyle='--', linewidth=2))

    grad_np = grad_field.cpu().numpy().T
    vmax = np.abs(grad_np).max()
    im2 = axes[1].imshow(grad_np, origin='lower', extent=domain_range,
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(r'$\partial \epsilon / \partial \theta$')
    axes[1].set_xlabel('X [m]'); axes[1].set_ylabel('Y [m]')
    plt.colorbar(im2, ax=axes[1])
    axes[1].add_patch(plt.Circle((circle_center_x, circle_center_y), radius,
                                  fill=False, color='black', linestyle='--', linewidth=2))

    im3 = axes[2].imshow(np.abs(grad_np), origin='lower', extent=domain_range, cmap='hot')
    axes[2].set_title(r'$|\partial \epsilon / \partial \theta|$')
    axes[2].set_xlabel('X [m]'); axes[2].set_ylabel('Y [m]')
    plt.colorbar(im3, ax=axes[2])
    axes[2].add_patch(plt.Circle((circle_center_x, circle_center_y), radius,
                                  fill=False, color='white', linestyle='--', linewidth=2))
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return grad_field, epsilon


def plot_gradient_vs_theta(theta_values=None, target_point=(100, 100),
                           circle_center_y=0.0, radius=0.2,
                           permittivity_value=4.0, grid_spacing=0.01,
                           domain_range=(-1.0, 1.0, -1.0, 1.0), device='cpu',
                           save_name='epsilon_gradient_vs_theta.png'):
    if theta_values is None:
        theta_values = np.linspace(-0.5, 0.5, 100)

    x_start, x_end, y_start, y_end = domain_range
    x = torch.linspace(x_start, x_end, int((x_end - x_start) / grid_spacing), device=device)
    y = torch.linspace(y_start, y_end, int((y_end - y_start) / grid_spacing), device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    aa_sharpness = 1.0 / grid_spacing
    cy = torch.tensor(circle_center_y, dtype=torch.float32, device=device)
    r = torch.tensor(radius, dtype=torch.float32, device=device)
    ti, tj = target_point

    epsilon_values, gradient_values = [], []
    for theta_val in theta_values:
        theta = torch.tensor(theta_val, dtype=torch.float32, device=device, requires_grad=True)
        dist = torch.sqrt((xx - theta)**2 + (yy - cy)**2)
        sdf = dist - r
        smooth_mask = torch.sigmoid(-sdf * aa_sharpness)
        epsilon = 1.0 * (1 - smooth_mask) + permittivity_value * smooth_mask
        eps_at_point = epsilon[ti, tj]
        epsilon_values.append(eps_at_point.item())
        eps_at_point.backward()
        gradient_values.append(theta.grad.item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_x, target_y = x[ti].item(), y[tj].item()

    axes[0].plot(theta_values, epsilon_values, 'b-', linewidth=2)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', label='background')
    axes[0].axhline(y=permittivity_value, color='orange', linestyle='--', label=f'circle eps={permittivity_value}')
    axes[0].set_xlabel(r'$\theta$'); axes[0].set_ylabel(r'$\epsilon$')
    axes[0].set_title(f'Epsilon at ({target_x:.2f}, {target_y:.2f}) vs theta')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(theta_values, gradient_values, 'r-', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_xlabel(r'$\theta$'); axes[1].set_ylabel(r'$\partial \epsilon / \partial \theta$')
    axes[1].set_title(f'Gradient at ({target_x:.2f}, {target_y:.2f}) vs theta')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return theta_values, epsilon_values, gradient_values


def plot_field_results(E_field, scene, polarization='TM', field_component='abs', save_name=None):
    if E_field is None:
        print("E_field is None"); return
    field_cpu = E_field.cpu()
    if field_component == 'real':
        field_to_plot = torch.real(field_cpu)
    elif field_component == 'imag':
        field_to_plot = torch.imag(field_cpu)
    elif field_component == 'abs':
        field_to_plot = torch.abs(field_cpu)
    else:
        raise ValueError("field_component must be 'real', 'imag', or 'abs'")
    label = 'Ez' if polarization == 'TM' else 'Hz'
    title = f"{field_component.capitalize()} of {label}"

    x_start, x_end, y_start, y_end = scene.domain_range
    aspect_ratio = (y_end - y_start) / (x_end - x_start)
    fig_width = 8
    plt.figure(figsize=(fig_width + 1.5, fig_width * aspect_ratio))
    plt.gca().set_facecolor('black'); plt.gcf().set_facecolor('black')
    im = plt.imshow(np.abs(field_to_plot.T.numpy()), cmap='jet', origin='lower',
                    extent=scene.domain_range, interpolation='bilinear')
    cbar = plt.colorbar(im); cbar.set_label(f"{label} amplitude", color='white')
    cbar.ax.tick_params(colors='white')
    plt.xticks(color='white'); plt.yticks(color='white')
    plt.xlabel('X [m]', color='white'); plt.ylabel('Y [m]', color='white')
    plt.contour(scene.x.cpu(), scene.y.cpu(), scene.permittivity.detach().cpu().T,
                levels=1, colors='w', linestyles='--')
    plt.title(title, color='white', pad=20)
    if save_name:
        save_path = os.path.join(FIGURES_DIR, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.show()
    return E_field


def plot_dE_dtheta_field(dE_dtheta, scene, save_name='dE_dtheta_field.png'):
    if dE_dtheta is None:
        print("dE_dtheta is None"); return None
    dE_np = dE_dtheta.cpu().numpy().T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (data, cmap, title) in zip(axes, [
        (np.real(dE_np), 'RdBu_r', r'Re($\partial E / \partial \theta$)'),
        (np.imag(dE_np), 'RdBu_r', r'Im($\partial E / \partial \theta$)'),
        (np.abs(dE_np), 'hot', r'$|\partial E / \partial \theta|$'),
    ]):
        vmax = np.abs(data).max() if cmap == 'RdBu_r' else None
        kwargs = dict(vmin=-vmax, vmax=vmax) if vmax else {}
        im = ax.imshow(data, origin='lower', extent=scene.domain_range, cmap=cmap, **kwargs)
        ax.set_title(title); ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        plt.colorbar(im, ax=ax)
        contour_color = 'black' if cmap == 'RdBu_r' else 'white'
        ax.contour(scene.x.cpu().numpy(), scene.y.cpu().numpy(),
                   scene.permittivity.detach().cpu().numpy().T,
                   levels=1, colors=contour_color, linestyles='--', linewidths=1)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return dE_dtheta


def compare_adjoint_vs_fd(deps_dtheta, E_field, dE_adj, dE_fd, scene,
                           save_name='adjoint_vs_fd.png'):
    deps_np = deps_dtheta.detach().cpu().numpy().T
    E_np = E_field.cpu().numpy().T
    dE_adj_np = dE_adj.cpu().numpy().T
    dE_fd_np = dE_fd.cpu().numpy().T
    eps_data = scene.permittivity.detach().cpu().numpy().T
    x_coords = scene.x.cpu().numpy()
    y_coords = scene.y.cpu().numpy()
    ext = scene.domain_range

    def add_contour(ax, color='white'):
        ax.contour(x_coords, y_coords, eps_data, levels=1,
                   colors=color, linestyles='--', linewidths=1)

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    # Row 1: d_eps/d_theta
    vmax = np.abs(deps_np).max()
    im = axes[0, 0].imshow(deps_np, origin='lower', extent=ext, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title(r'd$\epsilon$/d$\theta$'); plt.colorbar(im, ax=axes[0, 0]); add_contour(axes[0, 0], 'black')
    im = axes[0, 1].imshow(np.abs(deps_np), origin='lower', extent=ext, cmap='hot')
    axes[0, 1].set_title(r'|d$\epsilon$/d$\theta$|'); plt.colorbar(im, ax=axes[0, 1]); add_contour(axes[0, 1])
    im = axes[0, 2].imshow(eps_data, origin='lower', extent=ext, cmap='viridis')
    axes[0, 2].set_title(r'$\epsilon$ Field'); plt.colorbar(im, ax=axes[0, 2]); add_contour(axes[0, 2])

    # Row 2: E field
    for col, (data, title) in enumerate([
        (np.real(E_np), 'E Real'), (np.imag(E_np), 'E Imag'), (np.abs(E_np), 'E Magnitude')
    ]):
        cmap = 'hot' if col == 2 else 'RdBu_r'
        vmax = np.abs(data).max() if cmap == 'RdBu_r' else None
        kwargs = dict(vmin=-vmax, vmax=vmax) if vmax else {}
        im = axes[1, col].imshow(data, origin='lower', extent=ext, cmap=cmap, **kwargs)
        axes[1, col].set_title(title); plt.colorbar(im, ax=axes[1, col])
        add_contour(axes[1, col], 'black' if cmap == 'RdBu_r' else 'white')

    # Rows 3-4: Adjoint vs FD
    display_range = 0.3
    E_mag = np.abs(E_np)
    for row, (dE_np, label) in [(2, (dE_adj_np, 'Adjoint')), (3, (dE_fd_np, 'Finite Diff'))]:
        for col, (data, title) in enumerate([
            (np.real(dE_np), f'{label} Real'), (np.imag(dE_np), f'{label} Imag'),
            (np.real(np.conj(E_np) * dE_np) / (E_mag + 1e-10), f'{label} d|E|/dtheta')
        ]):
            im = axes[row, col].imshow(data, origin='lower', extent=ext,
                                       cmap='RdBu_r', vmin=-display_range, vmax=display_range)
            axes[row, col].set_title(title); plt.colorbar(im, ax=axes[row, col])
            add_contour(axes[row, col], 'black')

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
