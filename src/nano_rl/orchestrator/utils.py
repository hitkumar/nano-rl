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
    sampling_args = dict(sampling_config)
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        **sampling_config.extra_body,
        "return_token_ids": True,
        "top_k": -1,
        "min_p": 0.0,
    }
    # vLLM-specific params go in extra_body
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop(
        "repetition_penalty"
    )
    return sampling_args
