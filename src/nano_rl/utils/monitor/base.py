"""
Base monitor classe for metrics tracking
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Monitor(ABC):
    """
    Abstract base class for monitoring metrics
    """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics for a training step
        """
        pass

    def close(self) -> None:
        """
        Close the monitor. Override in subclasses that need to do any cleanup.
        """
        pass


class FilesystemMonitor(Monitor):
    """
    Monitor that stores metrics locally without using external services
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Args:
            output_dir: If provided, metrics will be saved to
                        output_dir/metrics.json on close()
        """
        self.output_dir = output_dir
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics for a training step
        """
        if step is not None:
            metrics["step"] = step
        self.history.append(metrics)

    def save_json(self) -> None:
        """
        Write the metrics to a file on close
        """
        if self.output_dir is None:
            return
        path = self.output_dir / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self) -> None:
        self.save_json()
