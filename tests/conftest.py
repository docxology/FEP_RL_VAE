"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_general_fep_rl: mark test as requiring general_FEP_RL package"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require general_FEP_RL if it's not available."""
    try:
        import general_FEP_RL
    except ImportError:
        skip_marker = pytest.mark.skip(reason="general_FEP_RL package not available")
        for item in items:
            if "general_FEP_RL" in str(item.fspath) or "test_models" in str(item.fspath):
                item.add_marker(skip_marker)
