import numpy as np
import torch

from witwin.maxwell.postprocess import compute_directivity


def _angular_grid(theta_points=721, phi_points=361):
    theta = np.linspace(0.0, np.pi, theta_points)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_points)
    return np.broadcast_arrays(theta[:, None], phi[None, :])


def _far_field_from_intensity(intensity, *, frequency=1.0e9, radius=3.0):
    theta, phi = _angular_grid(theta_points=intensity.shape[0], phi_points=intensity.shape[1])
    radius_grid = np.full(intensity.shape, radius, dtype=float)
    return {
        "theta": theta,
        "phi": phi,
        "frequency": frequency,
        "radius": radius_grid,
        "power_density": intensity / (radius_grid**2),
    }


def test_isotropic_radiator_has_unit_directivity():
    theta, phi = _angular_grid()
    intensity = np.full(theta.shape, 2.5, dtype=float)
    far_field = _far_field_from_intensity(intensity)

    result = compute_directivity(far_field)

    np.testing.assert_allclose(result["P_rad"], 4.0 * np.pi * 2.5, rtol=5e-6)
    np.testing.assert_allclose(result["directivity"], 1.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(result["D_max"], 1.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(result["D_max_db"], 0.0, atol=1e-5)


def test_hertzian_dipole_has_expected_peak_directivity():
    theta, phi = _angular_grid()
    intensity = np.sin(theta) ** 2 * np.ones_like(phi)
    far_field = _far_field_from_intensity(intensity)

    result = compute_directivity(far_field)

    assert np.isclose(result["D_max"], 1.5, rtol=1e-4, atol=1e-4)
    assert np.isclose(np.degrees(result["D_max_theta"]), 90.0, atol=1e-8)
    assert np.isclose(np.degrees(result["D_max_phi"]), 0.0, atol=1e-8)


def test_gain_equals_efficiency_times_directivity():
    theta, phi = _angular_grid()
    intensity = np.sin(theta) ** 2 * np.ones_like(phi)
    far_field = _far_field_from_intensity(intensity)

    directivity_only = compute_directivity(far_field)
    result = compute_directivity(far_field, input_power=2.0 * directivity_only["P_rad"])

    assert np.isclose(result["radiation_efficiency"], 0.5, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["gain"], 0.5 * directivity_only["directivity"], rtol=1e-6, atol=1e-6)
    assert np.isclose(result["G_max"], 0.5 * directivity_only["D_max"], rtol=1e-6, atol=1e-6)


def test_hertzian_dipole_half_power_beam_width_is_ninety_degrees():
    theta, phi = _angular_grid()
    intensity = np.sin(theta) ** 2 * np.ones_like(phi)
    far_field = _far_field_from_intensity(intensity)

    result = compute_directivity(far_field)

    assert np.isclose(result["beam_width_e_plane"], 90.0, atol=0.1)
    assert np.isclose(result["beam_width_h_plane"], 90.0, atol=0.1)


def test_directivity_keeps_torch_gradients():
    theta = torch.linspace(0.0, torch.pi, 181, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * torch.pi, 91, dtype=torch.float64)
    theta_grid, phi_grid = torch.broadcast_tensors(theta[:, None], phi[None, :])
    intensity = torch.full(theta_grid.shape, 2.5, dtype=torch.float64, requires_grad=True)
    radius = torch.full(theta_grid.shape, 3.0, dtype=torch.float64)
    far_field = {
        "theta": theta_grid,
        "phi": phi_grid,
        "frequency": 1.0e9,
        "radius": radius,
        "power_density": intensity / radius.square(),
    }

    result = compute_directivity(far_field, input_power=torch.tensor(20.0, dtype=torch.float64))
    loss = result["P_rad"] + result["directivity"].sum() + result["gain"].sum()
    loss.backward()

    assert isinstance(result["directivity"], torch.Tensor)
    assert intensity.grad is not None
    assert torch.all(torch.isfinite(intensity.grad))
