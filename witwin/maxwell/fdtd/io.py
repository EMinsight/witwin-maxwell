import numpy as np
import torch

from .postprocess import get_frequency_solution, get_material_permittivity


def _to_numpy_field(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_frequency_solution(solver, output_path):
    freq_solution = get_frequency_solution(solver)

    ex = np.pad(_to_numpy_field(freq_solution["Ex"]), ((0, 1), (0, 0), (0, 0)), mode="edge")
    ey = np.pad(_to_numpy_field(freq_solution["Ey"]), ((0, 0), (0, 1), (0, 0)), mode="edge")
    ez = np.pad(_to_numpy_field(freq_solution["Ez"]), ((0, 0), (0, 0), (0, 1)), mode="edge")

    field_data = np.stack(
        [ex.real, ex.imag, ey.real, ey.imag, ez.real, ez.imag],
        axis=-1,
    ).astype(np.float32)

    np.savez_compressed(
        output_path,
        field=field_data,
        permittivity=get_material_permittivity(solver).cpu().numpy(),
    )
    print(f"Frequency-domain solution saved to: {output_path}")
    print(f"Field data shape: {field_data.shape}")
