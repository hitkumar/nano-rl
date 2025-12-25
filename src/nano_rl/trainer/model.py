import time
from pathlib import Path

from datasets.formatting.torch_formatter import torch

import torch

import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from liger_kernel.transformers.auto_model import AutoLigerKernelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from nano_rl.trainer.config import ModelConfig, TokenizerConfig
from nano_rl.trainer.parallel_dims import ParallelDims
from nano_rl.utils.logger import get_logger
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def get_model(
    config: ModelConfig,
    device: torch.device = torch.device("meta"),
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    model_config = AutoConfig.from_pretrained(
        config.name,
        attn_implementation=config.attn,
        trust_remote_code=config.trust_remote_code,
    )
    # disable kv-cache for training
    model_config.use_cache = False
    with device:
        match config.impl:
            case "hf":
                model_cls = AutoModelForCausalLM
            case "liger_kernel":
                model_cls = AutoLigerKernelForCausalLM
            case _:
                raise ValueError(f"Unknown model implementation: {config.impl}")

        if device == torch.device("meta"):
            model = model_cls.from_config(
                model_config,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
            )
        else:
            model = model_cls.from_pretrained(
                config.name,
                config=model_config,
                trust_remote_code=config.trust_remote_code,
                dtype=dtype,
            )

    assert (
        model.lm_head.weight.dtype == dtype
    ), f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
    return model


def fix_model_post_empty(model: nn.Module) -> None:
    """
    Reinitializes the pos embedding buffers as they are not saved when model is saved.
    TODO: Does this work for Liger model class as well?
    """
    buffer_names = [name for name, _ in model.named_buffers()]
    # HF standard transformer model
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = model.model.rotary_emb
        inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
            rotary_emb.config, rotary_emb.inv_freq.device
        )
        rotary_emb.inv_freq.copy_(inv_freq)


def load_dcp_from_hf(model: nn.Module, config: ModelConfig) -> None:
    """
    Load model from Distributed checkpoint (DCP) from HF.
    Apprropriate weights are loaded from the checkpoint based on the current rank.
    """
    model.to_empty(device="cuda")
    dist.barrier()
    logger = get_logger()

    if not Path(config.name).exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    else:
        snapshot_path = Path(config.name)

    # with fsdp, this is only the parameters on this rank, not the full model
    logger.info(f"Loading weights using HF DCP from {snapshot_path}")
    load_dcp_start_time = time.perf_counter()
    state_dict = model.state_dict()
    if model.config.tie_word_embeddings:
        del state_dict["lm_head.weight"]

    dcp_load(
        state_dict,
        # as_posix() is basically str() but it will add a "/" at the end if it's not there which is useful for windows.
        storage_reader=HuggingFaceStorageReader(path=snapshot_path.as_posix()),
    )
    fix_model_post_empty(model)
    logger.debug(
        f"Loaded weights using HF DCP in {time.perf_counter() - load_dcp_start_time:.2f} seconds"
    )


def setup_model(config: ModelConfig, parallel_dims: ParallelDims) -> nn.Module:
    """Load model from HF"""
    model = get_model(
        config, device=torch.device("meta"), dtype=DTYPE_MAP[config.optimization_dtype]
    )
    # apply fsdp
    setup_fsdp(model, config, parallel_dims)

    # load weights from DCP
    load_dcp_from_hf(model, config)
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
