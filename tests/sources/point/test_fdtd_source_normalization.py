from types import SimpleNamespace

import numpy as np
import torch

from witwin.maxwell.fdtd.observers import get_observer_results


def test_source_normalization_applies_once_per_monitor():
    solver = SimpleNamespace(
        observers_enabled=True,
        complex_fields_enabled=False,
        _normalize_source=True,
        _source_time={"kind": "gaussian_pulse"},
        _observer_spectral_entries=[
            {
                "sample_count": 1,
                "frequency": 1.0,
                "source_dft_real": 1.0,
                "source_dft_imag": 0.0,
                "window_normalization": 0.5,
            }
        ],
        observers=[
            {
                "name": "shared_ex",
                "monitor_name": "shared",
                "kind": "point",
                "component": "Ex",
                "monitor_fields": ("Ex", "Ey"),
                "frequencies": (1.0,),
                "global_freq_indices": (0,),
                "group_component": "Ex",
                "group_offset": 0,
                "field_index": (0, 0, 0),
                "position": (0.0, 0.0, 0.0),
            },
            {
                "name": "shared_ey",
                "monitor_name": "shared",
                "kind": "point",
                "component": "Ey",
                "monitor_fields": ("Ex", "Ey"),
                "frequencies": (1.0,),
                "global_freq_indices": (0,),
                "group_component": "Ey",
                "group_offset": 0,
                "field_index": (0, 0, 0),
                "position": (0.0, 0.0, 0.0),
            },
        ],
        _point_observer_groups={
            "Ex": {
                "global_freq_indices": (0,),
                "freq_local_lookup": {0: 0},
                "real": torch.tensor([[3.0]], dtype=torch.float32),
                "imag": torch.tensor([[0.0]], dtype=torch.float32),
            },
            "Ey": {
                "global_freq_indices": (0,),
                "freq_local_lookup": {0: 0},
                "real": torch.tensor([[5.0]], dtype=torch.float32),
                "imag": torch.tensor([[0.0]], dtype=torch.float32),
            },
        },
        _plane_observer_groups={},
    )

    result = get_observer_results(solver)["shared"]

    # spectral_scale = 2 / 0.5 = 4, so the raw point values are 12 and 20.
    # source spectrum is also 4, so single-pass normalization should recover 3 and 5.
    assert result["Ex"] == np.complex128(3.0)
    assert result["Ey"] == np.complex128(5.0)
    assert result["components"]["Ex"] == np.complex128(3.0)
    assert result["components"]["Ey"] == np.complex128(5.0)
