"""
Filesystem based transport implementation
"""

from pathlib import Path

from nano_rl.transport.base import (
    MicroBatchReceiver,
    MicroBatchSender,
    TrainingBatchReceiver,
    TrainingBatchSender,
)
from nano_rl.transport.types import MicroBatch, TrainingBatch
from nano_rl.utils.pathing import get_rollout_dir, get_step_path, sync_wait_for_path

BATCH_FILE_TMP_NAME = "rollouts.bin.tmp"
BATCH_FILE_NAME = "rollouts.bin"


class FileSystemTrainingBatchSender(TrainingBatchSender):
    """Filesystem based training batch sender"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.rollout_dir = get_rollout_dir(output_dir)

    def send(self, batch: TrainingBatch) -> str:
        """Send a batch by writing it to disk"""
        step_path = get_step_path(self.rollout_dir, batch.step)
        step_path.mkdir(parents=True, exist_ok=True)

        buffer = self.encoder.encode(batch)
        tmp_path = step_path / BATCH_FILE_TMP_NAME
        with open(tmp_path, "wb") as f:
            f.write(buffer)

        # atomic rename so that receivers get consistent view of the data
        tmp_path.rename(step_path / BATCH_FILE_NAME)
        return str(step_path / BATCH_FILE_NAME)


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem based training batch receiver"""

    def __init__(self, output_dir: Path, current_step: int = 0):
        super().__init__()
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _get_batch_path(self) -> Path:
        return get_step_path(self.rollout_dir, self.current_step) / BATCH_FILE_NAME

    def wait(self) -> None:
        sync_wait_for_path(self._get_batch_path())

    def can_receive(self) -> bool:
        return self._get_batch_path().exists()

    def receive(self) -> TrainingBatch:
        with open(self._get_batch_path(), "rb") as f:
            batch = self.decoder.decode(f.read())

        self.current_step += 1
        return batch


class FileSystemMicroBatchSender(MicroBatchSender):
    """Filesystem based micro batch sender"""

    def __init__(self, output_dir: Path, dp_world_size: int, current_step: int = 0):
        super().__init__(output_dir, dp_world_size)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def send(self, micro_batch_grid: list[list[MicroBatch]], step: int) -> None:
        """Send micro batches to all DP ranks"""
        assert len(micro_batch_grid) == self.dp_world_size
        for micro_batch in micro_batch_grid:
            assert len(micro_batch) == len(
                micro_batch_grid[0]
            ), "All micro batch lists must be the same length"

        step_path = get_step_path(self.rollout_dir, self.current_step)
        step_path.mkdir(parents=True, exist_ok=True)

        for dp_rank in range(self.dp_world_size):
            micro_batch = micro_batch_grid[dp_rank]
            buffer = self.encoder.encode(micro_batch)
            tmp_path = step_path / f"rank_{dp_rank}.bin.tmp"
            with open(tmp_path, "wb") as f:
                f.write(buffer)

            # atomic rename so that receivers get consistent view of the data
            tmp_path.rename(step_path / f"rank_{dp_rank}.bin")

        self.current_step = step + 1


class FileSystemMicroBatchReceiver(MicroBatchReceiver):
    """Filesystem based micro batch receiver"""

    def __init__(self, output_dir: Path, dp_rank: int, current_step: int = 0):
        super().__init__(output_dir, dp_rank)
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _micro_batch_step_path(self):
        return (
            get_step_path(self.rollout_dir, self.current_step)
            / f"rank_{self.dp_rank}.bin"
        )

    def wait(self) -> None:
        sync_wait_for_path(self._micro_batch_step_path())

    def receive(self) -> list[MicroBatch]:
        with open(self._micro_batch_step_path(), "rb") as f:
            micro_batch = self.decoder.decode(f.read())
        self.current_step += 1
        return micro_batch
