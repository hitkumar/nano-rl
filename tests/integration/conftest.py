# Fixtures that are available to all integration tests

import pytest
from nano_rl.utils.logger import setup_logger


@pytest.fixture(scope="session", autouse=True)
def setup_test_logger():
    """Initialize logger once for all integration tests"""
    setup_logger(log_level="info", log_file=None)
