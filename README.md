# Witwin Maxwell

Witwin Maxwell is a PyTorch-native differentiable full-wave electromagnetic solver for Maxwell's equations. It exposes a single public workflow, `Scene -> Simulation -> Result`, while keeping inverse-design and optimization workloads inside standard PyTorch through `SceneModule`, `MaterialRegion`, and automatic backward support in the FDTD path.

The repository currently ships two solver backends:

- `FDFD`: frequency-domain finite-difference solver for single-frequency problems
- `FDTD`: GPU-first Yee-grid time-domain solver with monitor extraction, multi-frequency DFT sampling, and differentiable adjoint support

## Public API

The public model is intentionally small:

- `Scene`: domain, grid, boundaries, structures, sources, monitors, ports, and differentiable material regions
- `Simulation`: backend selection and runtime configuration through `Simulation.fdfd(...)` or `Simulation.fdtd(...)`
- `Result`: structured field access, material tensors, monitor payloads, plotting, stats, and save support

For module-style inverse-design workflows, define a `SceneModule`, implement `to_scene()`, and pass that module directly into `Simulation`. The runtime keeps the same `Scene -> Simulation -> Result` contract.

`Scene` stays declarative. Solver-sized Yee-grid coordinates, material tensors, and related runtime state are compiled during `Simulation.prepare()` / `Simulation.run()` and live on the internal solver scene rather than the public `Scene` object.

## Support Matrix

| Area | Currently supported | Notes |
| --- | --- | --- |
| Solvers | `Simulation.fdfd(...)`, `Simulation.fdtd(...)` | FDFD is single-frequency. FDTD supports time stepping and single- or multi-frequency DFT extraction. |
| Sources | `PointDipole`, `PlaneWave`, `GaussianBeam`, `ModeSource`, `TFSF` | `PlaneWave` / `GaussianBeam` support soft injection and `TFSF(...)`. `ModeSource` is still experimental. |
| Source time | `CW`, `GaussianPulse`, `RickerWavelet` | Shared waveform vocabulary across public source APIs. |
| Boundaries | `none`, `pml`, `periodic`, `bloch`, `pec`, `pmc` | Per-axis and per-face mixed layouts are available through `BoundarySpec.faces(...)`. |
| Materials | Isotropic `eps_r`, `mu_r`, `sigma_e`; `Debye`, `Drude`, `Lorentz`; `DiagonalTensor3`; `MaterialRegion` | `sigma_e` is the public frequency-domain conductivity path. `MaterialRegion` is the most direct differentiable design primitive. |
| Geometry | `Box`, `Sphere`, `Cylinder`, `Ellipsoid`, `Cone`, `Pyramid`, `Prism`, `Torus`, `HollowBox`, `Mesh` | Geometry and `Structure` primitives are re-exported through `witwin.maxwell`. |
| Monitors | `PointMonitor`, `PlaneMonitor`, `FluxMonitor`, `ModeMonitor` | Frequency selection is available through `Result.at(...)`. |
| Ports | `ModePort` | First-class modal port object; still experimental. |
| Results | `result.E`, `result.H`, `result.materials`, `Result.monitor(...)`, `Result.save(...)` | Structured field and material access stay torch-native. |
| Postprocess | Equivalent currents, Stratton-Chu propagation, near-to-far transform, directivity, bistatic RCS, S-parameters, modal overlap | Use `witwin.maxwell.postprocess`. |
| Differentiable workflows | `SceneModule`, `MaterialRegion`, supported trainable geometry parameters, FDTD adjoint backward | Public backward support currently targets trainable inputs that flow into the prepared-scene material tensors compiled from `Scene`. |

For the exhaustive user-visible capability inventory, see [`FEATURE_LIST.md`](FEATURE_LIST.md).

## Minimal Differentiable Example

The example below builds a tiny trainable design region, runs an FDTD simulation, reads a complex probe value, forms a scalar loss, and calls `loss.backward()` exactly like any other PyTorch program.

```python
import torch
import witwin.maxwell as mw


class TinyInverseDesign(mw.SceneModule):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros((1, 1, 1), device="cuda"))

    def to_scene(self):

        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.24, 0.24), (-0.24, 0.24), (-0.24, 0.24))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(0.0, 0.0, 0.06), size=(0.12, 0.12, 0.12)),
                density=torch.sigmoid(self.logits),
                eps_bounds=(1.0, 6.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.PointDipole(
                position=(0.0, 0.0, -0.06),
                polarization="Ez",
                width=0.04,
                source_time=mw.GaussianPulse(
                    frequency=1.0e9,
                    fwidth=0.25e9,
                    amplitude=50.0,
                ),
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.0, 0.0, 0.06), fields=("Ez",)))
        return scene


model = TinyInverseDesign().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

sim = mw.Simulation.fdtd(
    model,
    frequencies=[1.0e9],
    run_time=mw.TimeConfig(time_steps=32),
    spectral_sampler=mw.SpectralSampler(window="none"),
)

optimizer.zero_grad()
result = sim.run()

probe = result.monitor("probe")["data"]
target = torch.zeros((), dtype=probe.dtype, device=probe.device)
loss = torch.abs(probe - target) ** 2

loss.backward()
optimizer.step()

print("probe =", probe)
print("loss =", float(loss.detach().item()))
print("grad =", model.logits.grad)
```

This example is intentionally small, but it already uses the full public differentiable workflow:

`SceneModule -> Simulation.fdtd(...) -> Result -> loss.backward()`

The current public backward path is for trainable inputs that contribute to compiled material tensors, such as `MaterialRegion.density` and supported trainable geometry parameters. Parameters that only affect source placement or other non-material branches are not yet part of the public adjoint path.

## Installation

Core solver workflows are CUDA-first. A working NVIDIA GPU, a CUDA-compatible PyTorch build, and `slangtorch` are expected.

### 1. Create and activate an environment

```bash
conda create -n witwin2 python=3.11 -y
conda activate witwin2
```

### 2. Install a CUDA-enabled PyTorch build

Install the PyTorch build that matches your NVIDIA driver and CUDA stack first.

### 3. Install Maxwell

```bash
pip install -e .
```

Optional packages:

- `tidy3d` for `Scene.to_tidy3d(...)` and benchmark cross-validation workflows

## Development and Validation

Common local commands:

```bash
python -m pytest tests
python -m pytest tests/api/public/test_public_api.py tests/api/public/test_simulation_smoke.py
python -m pytest tests/boundaries/cpml/test_fdtd_cpml.py tests/monitors/observers/test_fdtd_observers.py
python -m pytest tests/gradients/test_fdtd_adjoint_bridge.py
python -m benchmark
python -m benchmark dipole_vacuum
python -m benchmark planewave_vacuum
```

Benchmark assets live under:

- `benchmark/scenes/`
- `benchmark/cache/`
- `benchmark/plots/`
- `benchmark/RESULTS.md`

## Current Notes

- Core FDFD and FDTD solver runs are GPU/CUDA-first.
- FDFD is a single-frequency public workflow.
- Prefer `DiagonalTensor3` for anisotropic materials. Full rotated/off-diagonal `Tensor3x3` support is not implemented yet.
- The public differentiable path currently focuses on trainable inputs that affect compiled material tensors.
- `ModeSource`, `ModeMonitor`, and `ModePort` are available, but they are still marked experimental.

## License

GPL-3.0-or-later. See `COPYING` for the full license text.
