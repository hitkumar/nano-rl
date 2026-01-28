"""
Training data ypes for transport between orchestrator and trainer
"""

import msgspec


class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    advantage: float | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    examples: list[TrainingSample]
    temperature: float
    step: int
    ckpt_step: int  # Policy step that generated this batch, used to measure how off policy our training is


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[int]  # true for completion tokens
    advantages: list[float]  # per token advantages
    inference_logprobs: list[float]
    temperature: float
    ckpt_step: int  # Policy step that generated this batch, used to measure how off policy our training is
