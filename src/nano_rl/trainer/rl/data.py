"""
Data loading for RL training.
Receives TrainingBatch from orchestrator via filesystem transport
"""

from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from nano_rl.trainer.rl.config import FakeDataLoaderConfig
from nano_rl.trainer.rl.packer import Packer, setup_packer
from nano_rl.trainer.world import get_world
from nano_rl.transport import MicroBatch, setup_micro_batch_receiver
from nano_rl.transport.config import TransportConfigType
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer


class TensorBatch(TypedDict):
    """Batch of training data for RL training as tensors"""

    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]

    # per token advantages from GRPO computation, usually constant across completion tokens of the same sample
    advantages: Float[Tensor, "batch seq"]

    # logprobs from inference time (the old policy). Used to compute the importance ratio.
    inference_logprobs: Float[Tensor, "batch seq"]

    # Sampling temperature used during rollout generation by vllm
    temperature: float

    # policy step used to generate this batch, used to measure how off policy our training is
    ckpt_step: int


class FakeDataLoader:
    """Fake debugging data"""

    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        self.seq_len = seq_len
        self.batch_size = config.batch_size
        self.step = 0
        self.dp_world_size = dp_world_size
        self.world = get_world()

        # compute the rank
        non_dp_world_size = self.world.world_size // self.dp_world_size
        self.rank = self.world.rank // non_dp_world_size

    def wait_for_batch(self) -> None:
        pass

    def get_batch(self) -> TensorBatch:
        # Use dp_rank in seed to ensure different data per rank
        seed = self.rank * 1000000 + self.step * 1000
        generator = torch.Generator().manual_seed(seed)
        self.step += 1

        input_ids = torch.randint(
            0, 32000, (self.batch_size, self.seq_len), generator=generator
        )
        position_ids = (
            torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
        )
        loss_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)

        # first 20% of seq is prompt, and needs to be masked
        prompt_len = self.seq_len // 5
        loss_mask[:, :prompt_len] = False

        advantages = torch.randn(self.batch_size, self.seq_len, generator=generator)
        inference_logprobs = (
            torch.randn(self.batch_size, self.seq_len, generator=generator) - 11.0
        )

        return TensorBatch(
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            advantages=advantages,
            inference_logprobs=inference_logprobs,
            temperature=1.0,
            ckpt_step=self.step,
        )


class DataLoader:
    """
    Loads micro batches written by packer. Master rank runs packer.
    """

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        seq_len: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        dp_world_size: int,
    ):
        self.world = get_world()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dp_world_size = dp_world_size

        # compute the rank
        non_dp_world_size = self.world.world_size // self.dp_world_size
        self.rank = self.world.rank // non_dp_world_size

        self.packer: Packer | None = None
        if self.world.is_master:
            self.packer = setup_packer(
                output_dir, dp_world_size, seq_len, tokenizer, config, start_step
            )

        # All ranks receive their own microbatches
        self.receiver = setup_micro_batch_receiver(
            output_dir, self.rank, start_step, config
        )

    def wait_for_batch(self) -> None:
        """
        Wait for next batch to be available.
        Master rank packs the data first, then all ranks wait for their files.
        """
        if self.packer is not None:
            self.packer.pack()
        self.receiver.wait()

    def get_batch(self) -> TensorBatch:
        """
        Get the next batch of training data.
        Returns a single TensorBatch with all micro batches stacked together.
        """
        micro_batches: list[MicroBatch] = self.receiver.receive()
        return self._collate_micro_batches(micro_batches)

    def _collate_micro_batches(self, micro_batches: list[MicroBatch]) -> TensorBatch:
        """
        Collate multiple MicroBatches into a single TensorBatch.
        Stacks all sequences into tensors of shape [num_micro_batches, seq_len].
        """
        return TensorBatch(
            input_ids=torch.tensor(
                [mb.input_ids for mb in micro_batches], dtype=torch.long
            ),
            position_ids=torch.tensor(
                [mb.position_ids for mb in micro_batches], dtype=torch.long
            ),
            loss_mask=torch.tensor(
                [mb.loss_mask for mb in micro_batches], dtype=torch.bool
            ),
            advantages=torch.tensor(
                [mb.advantages for mb in micro_batches], dtype=torch.float
            ),
            inference_logprobs=torch.tensor(
                [mb.inference_logprobs for mb in micro_batches], dtype=torch.float
            ),
            temperature=micro_batches[0].temperature,
            ckpt_step=micro_batches[0].ckpt_step,
        )
