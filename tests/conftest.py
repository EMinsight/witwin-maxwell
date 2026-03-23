"""Shared pytest configuration and fixtures."""


def pytest_addoption(parser):
    try:
        parser.addoption("--gpu", action="store_true", default=False,
                         help="Run GPU-only tests (FDFD/FDTD cross-validation)")
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --gpu flag is passed."""
    import torch
    if not config.getoption("--gpu") or not torch.cuda.is_available():
        skip = __import__('pytest').mark.skip(reason="needs --gpu flag and CUDA")
        for item in items:
            # Cross-validation tests require GPU
            if "test_fdfd_vs_fdtd" in str(item.fspath):
                item.add_marker(skip)
