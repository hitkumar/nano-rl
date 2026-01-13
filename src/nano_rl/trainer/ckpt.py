import shutil
from dataclasses import dataclass
from pathlib import Path

from nano_rl.trainer.config import CheckpointConfig
from nano_rl.trainer.weights import gather_weights_on_master, save_state_dict
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import get_logger
from nano_rl.utils.pathing import get_step_path, get_weights_dir
from torch import nn
from transformers import PreTrainedTokenizerBase


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class WeightCheckpointManager:
    """
    Save HF-compatible checkpoints for inference and eval. Not for resuming training.
    """

    def __init__(
        self,
        output_dir: Path,
        config: CheckpointConfig,
    ):
        self.weights_dir = get_weights_dir(output_dir)
        self.keep = config.keep
        self.config = config
        self.weights = config.weights
        self.logger = get_logger()
        self.world = get_world()
        self.saved_steps: list[int] = []

    def save(self, step: int, model: nn.Module, tokenizer: PreTrainedTokenizerBase):
        path = get_step_path(self.weights_dir, step)
        state_dict = gather_weights_on_master(model, self.world.is_master)

        if self.world.is_master:
            save_state_dict(
                state_dict, path, self.weights.save_format, self.weights.save_sharded
            )
            # save model config, tokenizer and tokenizer config
            model.config.save_pretrained(path)
            tokenizer.save_pretrained(path)
            self.saved_steps.append(step)
            self._maybe_clean()
            self.logger.info(f"Saved weights to {path}")
            (path / "STABLE").touch()

    def _maybe_clean(self) -> None:
        if self.keep is None or not self.world.is_master:
            return

        # fetch all but the last self.keep steps
        for step in self.saved_steps[: -self.keep]:
            path = get_step_path(self.weights_dir, step)
            if path.exists():
                shutil.rmtree(path)
                self.logger.info(f"Removed old ckpt weights {path}")

        self.saved_steps = self.saved_steps[-self.keep :]
