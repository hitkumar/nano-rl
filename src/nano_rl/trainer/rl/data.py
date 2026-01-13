"""
Data loading for RL training.
Receives TrainingBatch from orchestrator via filesystem transport
"""

from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int

from nano_rl.trainer.rl.config import FakeDataLoaderConfig
from nano_rl.trainer.world import get_world
from nano_rl.transport import setup_training_batch_receiver, TrainingBatch
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


class FakeDataLoader:
    """Fake debugging data"""

    def __init__(self, config: FakeDataLoaderConfig, seq_len: int):
        self.seq_len = seq_len
        self.batch_size = config.batch_size
        self.step = 0

    def wait_for_batch(self) -> None:
        pass

    def get_batch(self) -> TensorBatch:
        generator = torch.Generator().manual_seed(self.step)
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
        )


class DataLoader:
    """
    Loads training batches written by orchestrator using filesystem receive
    """

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        seq_len: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
    ):
        self.world = get_world()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.receiver = setup_training_batch_receiver(output_dir, start_step, config)

    def wait_for_batch(self) -> None:
        """
        Wait for next batch to be available.
        Blocks until orchestrator writes the batch file.
        """
        self.receiver.wait()

    def get_batch(self) -> TensorBatch:
        training_batch: TrainingBatch = self.receiver.receive()
        return self._batch_to_tensors(training_batch)

    def _batch_to_tensors(self, batch: TrainingBatch) -> TensorBatch:
        """
        Converts TrainingBatch from orchestrator to padded tensors that can be used for training
        TrainingSample contains variable-length lists:
        - prompt_ids: [1, 2, 3, ...]
        - completion_ids: [4, 5, 6, ...]
        - completion_logprobs: [-0.5, -0.3, ...]

        We need to:
        1. Concatenate prompt + completion
        2. Pad to same length
        3. Create appropriate masks
        """
        batch_size = len(batch.examples)
        # Use EOS as pad token if pad_token not set
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # Find max sequence length in batch (but cap at config seq_len)
        max_len = min(
            max(len(e.prompt_ids) + len(e.completion_ids) for e in batch.examples),
            self.seq_len,
        )

        # initialize tensors
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        advantages = torch.zeros(batch_size, max_len, dtype=torch.float)
        inference_logprobs = torch.zeros(batch_size, max_len, dtype=torch.float)

        for i, example in enumerate(batch.examples):
            prompt_ids = example.prompt_ids
            completion_ids = example.completion_ids
            completion_logprobs = example.completion_logprobs
            advantage = example.advantage if example.advantage is not None else 0.0

            total_len = len(prompt_ids) + len(completion_ids)
            if max_len < total_len:
                truncate_len = total_len - max_len
                completion_ids = completion_ids[:-truncate_len]
                completion_logprobs = completion_logprobs[:-truncate_len]
                total_len = len(prompt_ids) + len(completion_ids)

            prompt_len = len(prompt_ids)
            input_ids[i, :total_len] = torch.tensor(prompt_ids + completion_ids)
            position_ids[i, :total_len] = torch.arange(total_len)
            loss_mask[i, prompt_len:total_len] = True
            # same value for all tokens in completion, this is not ideal
            advantages[i, prompt_len:total_len] = advantage

            # Prompt tokens get 0 (they don't contribute to loss anyway)
            inference_logprobs[i, prompt_len:total_len] = torch.tensor(
                completion_logprobs
            )

        return TensorBatch(
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            advantages=advantages,
            inference_logprobs=inference_logprobs,
            temperature=batch.temperature,
        )
