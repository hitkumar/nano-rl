import torch
import torch.nn as nn

from nano_rl.trainer.config import ModelConfig, TokenizerConfig
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


def setup_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizer:
    """Load tokenizer from HF"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
