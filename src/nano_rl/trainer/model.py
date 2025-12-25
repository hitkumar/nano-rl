import torch
import torch.nn as nn

from nano_rl.trainer.config import ModelConfig, TokenizerConfig
from nano_rl.trainer.parallel_dims import ParallelDims
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def setup_model(config: ModelConfig) -> nn.Module:
    """Load model from HF"""
    model_config = AutoConfig.from_pretrained(
        config.name,
        attn_implementation=config.attn,
        trust_remote_code=config.trust_remote_code,
    )
    # disable kv-cache for training
    model_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        config=model_config,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    return model


def setup_fsdp(
    model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims
) -> nn.Module:
    """
    Wrap the model with FSDP for distributed training.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=DTYPE_MAP[config.reduce_dtype],
    )

    # we need the 2D grid here as fsdp does all reduce for gradients across the dp_replicate dimension
    # and across dp_shard_cp dimension, we do traditional FSDP (all gather during forward and reduce scatter during backward)
    if config.dp_replicate > 1:
        fsdp_mesh = parallel_dims.world_mesh["dp_replicate", "dp_shard_cp"]
    else:
        fsdp_mesh = parallel_dims.world_mesh["dp_shard_cp"]

    # Nested FSDP: per-layer sharding limits peak memory to one layer's weights.
    # Outer model wrap shards remaining params (embeddings, lm_head, final norm).
    for transformer_block in model.model.layers:
        fully_shard(
            transformer_block,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=config.reshard_after_forward,
        )

    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=config.reshard_after_forward,
    )
    return model


def setup_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizer:
    """Load tokenizer from HF"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
