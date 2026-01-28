from abc import ABC, abstractmethod
from pathlib import Path

import msgspec
from nano_rl.transport.types import MicroBatch, TrainingBatch


class TrainingBatchSender(ABC):
    """Base class for sending training batches from orchestrator to trainer"""

    def __init__(self, output_dir: Path):
        # self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir

    @abstractmethod
    def send(self, batch: TrainingBatch) -> str:
        """Returns the path where the batch was saved"""
        pass

    def close(self) -> None:
        pass


class TrainingBatchReceiver(ABC):
    """Base class for receiving training batches"""

    def __init__(self):
        self.decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    @abstractmethod
    def wait(self) -> None:
        pass

    @abstractmethod
    def can_receive(self) -> bool:
        pass

    @abstractmethod
    def receive(self) -> TrainingBatch:
        pass

    def close(self) -> None:
        pass


class MicroBatchSender(ABC):
    """Base class for sending micro batches from orchestrator to trainer"""

    def __init__(self, output_dir: Path, dp_world_size: int):
        # self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        self.dp_world_size = dp_world_size

    @abstractmethod
    def send(self, micro_batch_grid: list[list[MicroBatch]], step: int) -> None:
        """
        Send micro-batches for all DP ranks.
        micro_batch_grid[dp_rank] contains list of MicroBatch for that rank.
        """
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class MicroBatchReceiver(ABC):
    """Base class for receiving micro batches"""

    def __init__(self, output_dir: Path, dp_rank: int):
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.dp_rank = dp_rank

    @abstractmethod
    def wait(self) -> None:
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
