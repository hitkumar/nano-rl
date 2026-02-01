from pathlib import Path

import torch
from nano_rl.trainer.rl.broadcast.base import WeightBroadcast
from nano_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from nano_rl.trainer.rl.config import WeightBroadcastConfigType


def setup_weight_broadcast(
    output_dir: Path, config: WeightBroadcastConfigType, max_async_level: int
) -> WeightBroadcast:
    if config.type == "filesystem":
        # keep max_async_level + 1 ckpts by default
        return FileSystemWeightBroadcast(output_dir, config, max_async_level + 1)
    elif config.type == "nccl":
        from nano_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast

        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    else:
        raise ValueError(f"Unknown weight broadcast config type: {config.type}")
