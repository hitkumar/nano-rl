import pytest

from nano_rl.utils.logger import setup_logger


@pytest.fixture(autouse=True, scope="session")
def init_logger():
    setup_logger(log_level="warning", log_file=None)
