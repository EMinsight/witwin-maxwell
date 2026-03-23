import numpy as np

from witwin.maxwell.fdtd.evaluation import evaluate_reference_error, extract_aligned_reference


def test_extract_aligned_reference_crops_center_region():
    reference = {
        "field": np.arange(7 * 7 * 7, dtype=np.float32).reshape(7, 7, 7),
        "x": np.linspace(-3.0, 3.0, 7),
        "y": np.linspace(-3.0, 3.0, 7),
        "z": np.linspace(-3.0, 3.0, 7),
    }
    target = {
        "field": np.zeros((3, 3, 3), dtype=np.float32),
        "x": np.linspace(-1.0, 1.0, 3),
        "y": np.linspace(-1.0, 1.0, 3),
        "z": np.linspace(-1.0, 1.0, 3),
    }

    cropped = extract_aligned_reference(reference, target)

    np.testing.assert_array_equal(cropped, reference["field"][2:5, 2:5, 2:5])


def test_evaluate_reference_error_zero_for_matching_fields():
    field = np.ones((9, 9, 9), dtype=np.float32)
    target = {
        "field": field.copy(),
        "x": np.linspace(-0.4, 0.4, 9),
        "y": np.linspace(-0.4, 0.4, 9),
        "z": np.linspace(-0.4, 0.4, 9),
    }
    reference = {
        "field": np.pad(field, 2, mode="constant", constant_values=1.0),
        "x": np.linspace(-0.6, 0.6, 13),
        "y": np.linspace(-0.6, 0.6, 13),
        "z": np.linspace(-0.6, 0.6, 13),
    }

    metrics = evaluate_reference_error(target, reference, pml_thickness=2)

    assert metrics["reference_rel_l2"] == 0.0
    assert metrics["reference_rel_linf"] == 0.0
    assert metrics["axis_rel_l2"] == 0.0
    assert metrics["max_abs_diff"] == 0.0
