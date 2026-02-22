""" Orchestrator utils """

from typing import Any

from nano_rl.orchestrator.config import SamplingConfig


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
