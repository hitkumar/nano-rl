import json
import warnings
from pathlib import Path
from typing import cast, Literal

import torch
import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from nano_rl.utils.logger import get_logger
from safetensors.torch import save_file
from torch import nn, Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """
    Gather FSDP-sharded weights to the master rank for saving
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # full_tensor is a sync operation, after this line all ranks will have the full tensor
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                # strip fsdp related info from key name
                fqn = next(iter(get_fqns(model, key)))
                # Finish CPU transfer of this tensor before moving to next.
                cpu_state[fqn] = value.to("cpu", non_blocking=False)

        dist.barrier()
    return cpu_state


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: only supports safetensors save format, support torch as well.
    if save_sharded:
        split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern="model{suffix}.safetensors"
        )
        for shard_file, tensors in split.filename_to_tensors.items():
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                del state_dict[tensor]
            save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
        del state_dict
        if split.is_sharded:
            index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
            with open(save_dir / SAFE_WEIGHTS_INDEX_NAME, "w") as f:
                json.dump(index, f, indent=2)
    else:
        if save_format == "safetensors":
            save_file(
                state_dict, save_dir / SAFE_WEIGHTS_NAME, metadata={"format": "pt"}
            )
        else:
            torch.save(state_dict, save_dir / WEIGHTS_NAME)
