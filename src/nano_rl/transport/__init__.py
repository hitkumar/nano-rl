"""Transport layer for orchestrator-trainer communication"""

from pathlib import Path

from multiprocess.sharedctypes import Value
from nano_rl.transport.base import (
    MicroBatchReceiver,
    MicroBatchSender,
    TrainingBatchReceiver,
    TrainingBatchSender,
)
from nano_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from nano_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from nano_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample


def setup_training_batch_sender(
    output_dir: Path, transport_type: TransportConfigType = FileSystemTransportConfig()
) -> TrainingBatchSender:
    if transport_type.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)

    raise ValueError(f"Invalid transport type passed: {transport_type.type}")


def setup_training_batch_receiver(
    output_dir: Path,
    current_step: int,
    transport_type: TransportConfigType = FileSystemTransportConfig(),
) -> TrainingBatchReceiver:
    if transport_type.type == "filesystem":
        return FileSystemTrainingBatchReceiver(output_dir, current_step)

    raise ValueError(f"Invalid transport type passed: {transport_type.type}")


def setup_micro_batch_sender(
    output_dir: Path,
    dp_world_size: int,
    start_step: int = 0,
    transport_type: TransportConfigType = FileSystemTransportConfig(),
) -> MicroBatchSender:
    if transport_type.type == "filesystem":
        return FileSystemMicroBatchSender(output_dir, dp_world_size, start_step)
    raise ValueError(f"Invalid transport type: {transport_type.type}")


def setup_micro_batch_receiver(
    output_dir: Path,
    dp_rank: int,
    start_step: int = 0,
    transport_type: TransportConfigType = FileSystemTransportConfig(),
) -> MicroBatchReceiver:
    if transport_type.type == "filesystem":
        return FileSystemMicroBatchReceiver(output_dir, dp_rank, start_step)
    raise ValueError(f"Invalid transport type: {transport_type.type}")


# modules imported by wildcard import.
__all__ = [
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "TrainingBatchReceiver",
    "TrainingBatchSender",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TransportConfigType",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
