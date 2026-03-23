import numpy as np
import torch


def get_field_raw(solver):
    if solver.E_field is None:
        return None
    ex, ey, ez = solver.E_field
    return {
        "Ex": ex.cpu().numpy().astype(np.complex64),
        "Ey": ey.cpu().numpy().astype(np.complex64),
        "Ez": ez.cpu().numpy().astype(np.complex64),
    }


def interpolate_yee_to_center(solver):
    ex, ey, ez = solver.E_field
    if ex.shape == ey.shape == ez.shape:
        field_data = ex
    else:
        ex_int = 0.5 * (ex[:, :-1, :] + ex[:, 1:, :])
        ex_int = 0.5 * (ex_int[:, :, :-1] + ex_int[:, :, 1:])
        ey_int = 0.5 * (ey[:-1, :, :] + ey[1:, :, :])
        ey_int = 0.5 * (ey_int[:, :, :-1] + ey_int[:, :, 1:])
        ez_temp = 0.5 * (ez[:-1, :, :] + ez[1:, :, :])
        ez_int = 0.5 * (ez_temp[:, :-1, :] + ez_temp[:, 1:, :])
        field_data = torch.sqrt(torch.abs(ex_int) ** 2 + torch.abs(ey_int) ** 2 + torch.abs(ez_int) ** 2)

    dx, dy, dz = solver.scene.grid_spacing
    x_c = (solver.scene.x[:-1] + dx / 2.0).cpu().numpy()
    y_c = (solver.scene.y[:-1] + dy / 2.0).cpu().numpy()
    z_c = (solver.scene.z[:-1] + dz / 2.0).cpu().numpy()
    return field_data.cpu().numpy(), x_c, y_c, z_c
