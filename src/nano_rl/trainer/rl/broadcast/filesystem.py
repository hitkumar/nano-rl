import shutil
import time
from pathlib import Path
from typing import Literal

import torch.nn as nn
from nano_rl.trainer.rl.broadcast.base import WeightBroadcast
from nano_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from nano_rl.trainer.weights import gather_weights_on_master, save_state_dict
from nano_rl.trainer.world import get_world
from nano_rl.utils.pathing import get_broadcasts_dir, get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcasts weights to inference engine via filesystem"""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, keep: int
    ):
        super().__init__(output_dir)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded
        self.keep = keep
        self.world = get_world()
        self.broadcasts_dir = get_broadcasts_dir(output_dir)
        self.saved_steps: list[int] = []
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={self.save_sharded}, keep={self.keep})"
        )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Saves the weights to filesystem and notifies orch, follows similar logic as ckpt.save"""
        start_time = time.perf_counter()

        # Gather weights from all FSDP ranks to master
        state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
        if self.world.is_master:
            save_dir = get_step_path(self.broadcasts_dir, step)
            save_dir.mkdir(exist_ok=True, parents=True)

            save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded)

            # notify orch after fully saving state dict
            stable_file = save_dir / "STABLE"
            stable_file.touch()

            self.saved_steps.append(step)
            self._maybe_clean()
            self.logger.info(
                f"Weights broadcasted to {save_dir} in {time.perf_counter() - start_time:.2f}s"
            )

    def _maybe_clean(self):
        """Removes old broadcast directories, keeping only the most recent ones"""
        if self.keep is None or not self.world.is_master:
            return

        for step in self.saved_steps[: -self.keep]:
            path = get_step_path(self.broadcasts_dir, step)
            if path.exists():
                shutil.rmtree(path)
                self.logger.debug(f"Removed old broadcast path: {path}")

        self.saved_steps = self.saved_steps[-self.keep :]
