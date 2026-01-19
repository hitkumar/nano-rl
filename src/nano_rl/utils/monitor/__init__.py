from pathlib import Path

from nano_rl.utils.monitor.base import FilesystemMonitor, Monitor
from nano_rl.utils.pydantic_config import BaseSettings

__all__ = [
    "setup_monitor",
    "get_monitor",
]

# Singleton initiliazed once per training run

_MONITOR: Monitor | None = None


def setup_monitor(
    output_dir: Path | None = None,
    run_config: BaseSettings | None = None,
) -> Monitor:
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError(
            "Monitor already initialized. Call setup_monitor() only once."
        )

    # run_config is unused here
    _MONITOR = FilesystemMonitor(output_dir)
    return _MONITOR


def get_monitor() -> Monitor:
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError(
            "Monitor not initialized. Please call setup_monitor() first."
        )
    return _MONITOR
