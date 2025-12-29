"""
Filesystem based transport implementation
"""

from pathlib import Path

from nano_rl.transport.base import TrainingBatchReceiver, TrainingBatchSender
from nano_rl.transport.types import TrainingBatch
from nano_rl.utils.pathing import get_rollout_dir, get_step_path, sync_wait_for_path

BATCH_FILE_TMP_NAME = "rollouts.bin.tmp"
BATCH_FILE_NAME = "rollouts.bin"


class FileSystemTrainingBatchSender(TrainingBatchSender):
    """Filesystem based training batch sender"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.rollout_dir = get_rollout_dir(output_dir)

    def send(self, batch: TrainingBatch) -> None:
        """Send a batch by writing it to disk"""
        step_path = get_step_path(self.rollout_dir, batch.step)
        step_path.mkdir(parents=True, exist_ok=True)

        buffer = self.encoder.encode(batch)
        tmp_path = step_path / BATCH_FILE_TMP_NAME
        with open(tmp_path, "wb") as f:
            f.write(buffer)

        # atomic rename so that receivers get consistent view of the data
        tmp_path.rename(step_path / BATCH_FILE_NAME)


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem based training batch receiver"""

    def __init__(self, output_dir: Path, current_step: int = 0):
        super().__init__()
        self.rollout_dir = get_rollout_dir(output_dir)
        self.current_step = current_step

    def _get_batch_path(self) -> Path:
        return get_step_path(self.rollout_dir, self.current_step) / BATCH_FILE_NAME

    def wait(self) -> None:
        path = self._get_batch_path()
        sync_wait_for_path(path)

    def can_receive(self) -> bool:
        return self._get_batch_path().exists()

    def receive(self) -> TrainingBatch:
        with open(self._get_batch_path(), "rb") as f:
            batch = self.decoder.decode(f.read())

        self.current_step += 1
        return batch
