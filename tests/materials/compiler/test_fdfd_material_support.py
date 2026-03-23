import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.scene import prepare_scene


def _build_cuda_scene(*, material):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, 0.0), size=(0.3, 0.3, 0.3)),
            material=material,
        )
    )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_operator_uses_axis_aligned_anisotropic_permittivity_components():
    cp = pytest.importorskip("cupy")
    frequency = 2.0e9
    scene = _build_cuda_scene(
        material=mw.Material(
            eps_r=1.0,
            epsilon_tensor=mw.DiagonalTensor3(2.0, 4.0, 8.0),
        )
    )

    solver = mw.Simulation.fdfd(scene, frequency=frequency).prepare().solver
    compiled_scene = prepare_scene(scene)
    solver.material_eps_components, solver.material_mu_components = compiled_scene.compile_material_components(
        frequency=frequency
    )
    diagonal = cp.asnumpy(solver._build_matrix_gpu_yee_3d().diagonal())

    n_ex = compiled_scene.N_ex
    n_ey = compiled_scene.N_ey
    ex_mean = diagonal[:n_ex].real.mean()
    ey_mean = diagonal[n_ex:n_ex + n_ey].real.mean()
    ez_mean = diagonal[n_ex + n_ey:].real.mean()

    assert ex_mean < ey_mean < ez_mean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_operator_uses_component_specific_sigma_tensor_conductivity():
    cp = pytest.importorskip("cupy")
    frequency = 2.0e9
    scene = _build_cuda_scene(
        material=mw.Material(
            eps_r=2.0,
            sigma_e_tensor=mw.DiagonalTensor3(0.0, 0.0, 5.0),
        )
    )

    solver = mw.Simulation.fdfd(scene, frequency=frequency).prepare().solver
    compiled_scene = prepare_scene(scene)
    solver.material_eps_components, solver.material_mu_components = compiled_scene.compile_material_components(
        frequency=frequency
    )
    diagonal = cp.asnumpy(solver._build_matrix_gpu_yee_3d().diagonal())

    n_ex = compiled_scene.N_ex
    n_ey = compiled_scene.N_ey
    ex_imag = np.abs(diagonal[:n_ex].imag).mean()
    ey_imag = np.abs(diagonal[n_ex:n_ex + n_ey].imag).mean()
    ez_imag = np.abs(diagonal[n_ex + n_ey:].imag).mean()

    assert ex_imag < 1.0e-6
    assert ey_imag < 1.0e-6
    assert ez_imag > 1.0e3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_rejects_static_magnetic_response():
    scene = _build_cuda_scene(material=mw.Material(eps_r=2.0, mu_r=1.5))

    with pytest.raises(NotImplementedError, match="magnetic media and magnetic dispersion"):
        mw.Simulation.fdfd(scene, frequency=1.0e9).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDFD backend")
def test_fdfd_rejects_kerr_media():
    scene = _build_cuda_scene(material=mw.Material(eps_r=2.0, kerr_chi3=1.0e-10))

    with pytest.raises(NotImplementedError, match="Kerr nonlinear media"):
        mw.Simulation.fdfd(scene, frequency=1.0e9).prepare()
