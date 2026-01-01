""" Orchestrator utils """

import asyncio
from contextlib import nullcontext
from typing import Any

from nano_rl.orchestrator.config import SamplingConfig

_SEMAPHORE: asyncio.Semaphore | None = None


async def set_semaphore(limit: int | None) -> None:
    global _SEMAPHORE
    if limit is not None:
        _SEMAPHORE = asyncio.Semaphore(limit)


async def get_semaphore():
    """Get global semaphore or null context if not set"""
    if _SEMAPHORE is not None:
        return _SEMAPHORE
    return nullcontext()


def get_sampling_args(sampling_config: SamplingConfig) -> dict[str, Any]:
    """Convert sampling config to dict for verifiers"""
    args = {
        "temperature": sampling_config.temperature,
        "repetition_penalty": sampling_config.repetition_penalty,
        "min_tokens": sampling_config.min_tokens,
        "extra_body": sampling_config.extra_body,
    }
    if sampling_config.max_tokens is not None:
        args["max_tokens"] = sampling_config.max_tokens
    if sampling_config.top_p is not None:
        args["top_p"] = sampling_config.top_p
    if sampling_config.top_k is not None:
        args["top_k"] = sampling_config.top_k
    if sampling_config.seed is not None:
        args["seed"] = sampling_config.seed
    return args
