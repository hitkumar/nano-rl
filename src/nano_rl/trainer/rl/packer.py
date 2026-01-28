"""
Packer: Splits TrainingBatch into MicroBatches distributed across DP ranks.

Data flow:
  Orchestrator -> TrainingBatch -> Packer -> MicroBatch per DP rank -> Trainer
"""

from pathlib import Path

from nano_rl.transport import (
    MicroBatchSender,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
    TrainingBatchReceiver,
)
from nano_rl.transport.config import TransportConfigType
from nano_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample
from nano_rl.utils.logger import get_logger
from transformers import PreTrainedTokenizer


def prepare_sample(
    training_sample: TrainingSample, temperature: float, ckpt_step: int
) -> MicroBatch:
    """
    Converts one training sample into a micro batch.
    This is before packing - each sample becomes its own MicroBatch initially.
    """
    prompt_ids = training_sample.prompt_ids
    completion_ids = training_sample.completion_ids
    completion_logprobs = training_sample.completion_logprobs
    advantage = (
        training_sample.advantage if training_sample.advantage is not None else 0.0
    )

    input_ids = prompt_ids + completion_ids
    # Position IDs: 0, 1, 2, ... for the full sequence
    position_ids = list(range(len(input_ids)))
    prompt_len = len(prompt_ids)
    completion_len = len(completion_ids)
    loss_mask = [0] * prompt_len + [1] * completion_len
    advantages = [0] * prompt_len + [advantage] * completion_len
    inference_logprobs = [0] * prompt_len + completion_logprobs

    return MicroBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        loss_mask=loss_mask,
        advantages=advantages,
        inference_logprobs=inference_logprobs,
        temperature=temperature,
        ckpt_step=ckpt_step,
    )


def pack_samples_into_micro_batches(
    samples: list[MicroBatch], seq_len: int, pad_int: int
) -> list[MicroBatch]:
    """
    Packs samples into micro batches.
    Input samples is 1:1 with the training batch. Packing is done so that GPUs are efficiently used using first fit decreasing algorithm.
    """
    if not samples:
        return []

    # sort by decreasing length so that we can pack efficiently
    sorted_samples = sorted(samples, key=lambda x: len(x.input_ids), reverse=True)
    packed_batches = []
    for sample in sorted_samples:
        sample_len = len(sample.input_ids)
        if sample_len > seq_len:
            # Truncate to seq_len
            sample = MicroBatch(
                input_ids=sample.input_ids[:seq_len],
                position_ids=sample.position_ids[:seq_len],
                loss_mask=sample.loss_mask[:seq_len],
                advantages=sample.advantages[:seq_len],
                inference_logprobs=sample.inference_logprobs[:seq_len],
                temperature=sample.temperature,
                ckpt_step=sample.ckpt_step,
            )
            sample_len = seq_len

        placed = False
        for packed_batch in packed_batches:
            batch_len = len(packed_batch.input_ids)
            if batch_len + sample_len <= seq_len:
                packed_batch.input_ids.extend(sample.input_ids)
                packed_batch.position_ids.extend(sample.position_ids)
                packed_batch.loss_mask.extend(sample.loss_mask)
                packed_batch.advantages.extend(sample.advantages)
                packed_batch.inference_logprobs.extend(sample.inference_logprobs)
                placed = True
                break

        if not placed:
            packed_batches.append(
                MicroBatch(
                    input_ids=list(sample.input_ids),
                    position_ids=list(sample.position_ids),
                    loss_mask=list(sample.loss_mask),
                    advantages=list(sample.advantages),
                    inference_logprobs=list(sample.inference_logprobs),
                    temperature=sample.temperature,
                    ckpt_step=sample.ckpt_step,
                )
            )

    for batch in packed_batches:
        batch_len = len(batch.input_ids)
        if batch_len < seq_len:
            pad_len = seq_len - batch_len
            batch.input_ids.extend([pad_int] * pad_len)
            batch.position_ids.extend([0] * pad_len)
            batch.loss_mask.extend([0] * pad_len)
            batch.advantages.extend([0] * pad_len)
            batch.inference_logprobs.extend([0] * pad_len)

    return packed_batches


def prepare_batch(
    training_batch: TrainingBatch, dp_world_size: int, seq_len: int, pad_id: int
) -> list[list[MicroBatch]]:
    """
    Convert TrainingBatch into a grid of MicroBatches for each DP rank.

    Args:
        training_batch: Batch from orchestrator with training samples
        dp_world_size: Number of data parallel ranks
        seq_len: Target sequence length
        pad_id: Padding token ID

    Returns:
        micro_batch_grid[dp_rank] = list of MicroBatches for that rank
    """
    samples = [
        prepare_sample(sample, training_batch.temperature, training_batch.ckpt_step)
        for sample in training_batch.examples
    ]
    packed_samples = pack_samples_into_micro_batches(samples, seq_len, pad_id)
    if len(packed_samples) % dp_world_size != 0:
        get_logger().warning(
            f"packed_samples count ({len(packed_samples)}) is not divisible by dp_world_size ({dp_world_size}), "
            f"adding {dp_world_size - len(packed_samples) % dp_world_size} dummy batch(es)"
        )
    while len(packed_samples) % dp_world_size != 0:
        # Create a dummy batch with all padding (zero advantage = no gradient contribution)
        dummy = MicroBatch(
            input_ids=[pad_id] * seq_len,
            position_ids=list(range(seq_len)),
            loss_mask=[0] * seq_len,  # All masked = no loss
            advantages=[0.0] * seq_len,
            inference_logprobs=[0.0] * seq_len,
            temperature=training_batch.temperature,
            ckpt_step=training_batch.ckpt_step,
        )
        packed_samples.append(dummy)

    micro_batch_grid: list[list[MicroBatch]] = [[] for _ in range(dp_world_size)]
    for i, packed_batch in enumerate(packed_samples):
        micro_batch_grid[i % dp_world_size].append(packed_batch)

    return micro_batch_grid


class Packer:
    """
    Receives TrainingBatch from orchestrator, packs into MicroBatches,
    and sends to per-rank files for trainer consumption.

    Only runs on master rank (rank 0).
    """

    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        tokenizer: PreTrainedTokenizer,
        receiver: TrainingBatchReceiver,
        sender: MicroBatchSender,
    ):
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.receiver = receiver
        self.sender = sender
        self.logger = get_logger()

    def pack(self) -> None:
        """
        Wait for TrainingBatch, pack into MicroBatches, send to ranks.
        Called once per training step by master rank.
        """
        self.receiver.wait()
        training_batch: TrainingBatch = self.receiver.receive()
        self.logger.debug(
            f"Packer received batch step={training_batch.step} "
            f"with {len(training_batch.examples)} examples"
        )
        micro_batch_grid = prepare_batch(
            training_batch, self.dp_world_size, self.seq_len, self.pad_id
        )
        batches_per_rank = len(micro_batch_grid[0])
        self.logger.debug(f"Sending {batches_per_rank} batches per rank")
        self.sender.send(micro_batch_grid, training_batch.step)


def setup_packer(
    output_dir: Path,
    dp_world_size: int,
    seq_len: int,
    tokenzer: PreTrainedTokenizer,
    transport_config: TransportConfigType,
    start_step: int = 0,
) -> Packer:
    receiver: TrainingBatchReceiver = setup_training_batch_receiver(
        output_dir, start_step, transport_config
    )
    sender: MicroBatchSender = setup_micro_batch_sender(
        output_dir, dp_world_size, start_step, transport_config
    )
    return Packer(
        dp_world_size=dp_world_size,
        seq_len=seq_len,
        tokenizer=tokenzer,
        receiver=receiver,
        sender=sender,
    )
