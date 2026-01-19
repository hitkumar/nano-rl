"""
Performance counter for tracking throughput and MFU with a rolling window.
MFU (Model FLOPs Utilization) measures what percentage of theoretical GPU compute
you're actually using. This helps identify bottlenecks:
- Low MFU (<30%) → memory bound or poor parallelization
- High MFU (>50%) → compute bound, good utilization

Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/utils.py
"""

import time

import torch
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import get_logger
from torch import nn
from transformers import PretrainedConfig


class PerfCounter:
    """
    Tracks performance with a rolling window for stable MFU estimates
    Rolling window smoothens step by step variations in MFU to give a more stable estimate
    """

    def __init__(self, model: nn.Module, seq_len: int, window_size: int = 10):
        self.window_size = window_size
        self.tokens: list[int] = []  # Token counts per step
        self.times: list[float] = []  # time per steps
        self.model = model

        self._world = get_world()
        self._logger = get_logger()

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.device("cuda"))
            self.gpu_peak_flops = self._get_peak_flops(device_name)
        else:
            self.gpu_peak_flops = 0

        # Calculate FLOPs per token for this model
        self.num_flop_per_token = self._get_num_flops_per_token(model.config, seq_len)

    def count_tokens(self, tokens: int) -> None:
        """Record token count at current time for throughput calculation"""
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())

        # Maintain rolling window
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_sec(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])

    def get_mfu(self) -> float | None:
        """
        Calculate Model FLOPs Utilization as a percentage.
        MFU = (actual_flops / peak_possible_flops) * 100
        """
        tokens_per_second = self.get_tokens_per_sec()
        if tokens_per_second is None:
            return None
        actual_flops = tokens_per_second * self.num_flop_per_token
        peak_flops = self.gpu_peak_flops * self._world.world_size

        return 100 * actual_flops / peak_flops

    def _get_num_flops_per_token(self, config: PretrainedConfig, seq_len: int):
        """
        Flops per token for full fine tuning

        We calculate flops for attention computation and other matrix multiplications separately
        For each matmul:
        - Forward: 2 FLOPs per multiply-add
        - Backward: 4 FLOPs (gradient w.r.t. input + gradient w.r.t. weights)
        """
        l, h, q, t = (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            seq_len,
        )

        # 2 matmul in fwd, 4 matmul in bwd, 2 flops per matmul
        attention_flops = 12 * l * h * q * t
        # 1 matmul in fwd, 2 matmul in bwd, 2 flops per matmul
        matmul_flops = 6 * self.get_active_mm_params(config)
        return attention_flops + matmul_flops

    @staticmethod
    def get_active_mm_params(config: PretrainedConfig) -> int:
        """
        Get the number of params involved in matrix multiply for a token
        Counts params in
        - attention calculations
        - MLP calculations
        - LM head projection

        Assumes embedding params are shared with lm_head.
        """
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        # number of blocks in the model
        num_hidden_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(
            config, "num_key_value_heads", num_attention_heads
        )
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        q_params = num_hidden_layers * hidden_size * hidden_size
        # Accounting for use of GQA attention
        kv_params = num_hidden_layers * hidden_size * num_key_value_heads * head_dim
        o_params = num_hidden_layers * hidden_size * hidden_size
        mlp_params = num_hidden_layers * 3 * hidden_size * intermediate_size
        lm_heads_params = hidden_size * vocab_size

        return q_params + kv_params + o_params + mlp_params + lm_heads_params

    def _get_peak_flops(self, device_name: str) -> float:
        """
        Get peak BF16 FLOPs for common GPU types.

        These are theoretical peaks without sparsity.
        Source: NVIDIA datasheets
        """
        if "A100" in device_name:
            # A100 SXM: 312 TFLOPS BF16
            return 312e12
        if "H100" in device_name or "H200" in device_name:
            if "NVL" in device_name:
                return 835e12
            elif "PCIe" in device_name:
                return 756e12
            else:  # H100 SXM
                return 989e12
        if "B200" in device_name:
            return 2.25e15
        else:
            if self._world.is_master:
                self._logger.warning(
                    f"Peak FLOPS undefined for `{device_name}`. "
                    "Falling back to A100 (312 TFLOPS)"
                )
            return 312e12


# Singleton pattern so that we have one perfcounter per training run

_PERF_COUNTER: PerfCounter | None = None


def get_perf_counter(
    model: nn.Module, seq_len: int, window_size: int = 10
) -> PerfCounter:
    global _PERF_COUNTER
    if _PERF_COUNTER is None:
        _PERF_COUNTER = PerfCounter(model, seq_len, window_size)
    return _PERF_COUNTER


def reset_perf_counter() -> None:
    global _PERF_COUNTER
    _PERF_COUNTER = None
