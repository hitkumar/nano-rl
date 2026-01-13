from datetime import timedelta

import torch
import torch.distributed as dist
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import get_logger

DEFAULT_TIMEOUT = timedelta(seconds=300)


def setup_torch_distributed(
    timeout: timedelta = DEFAULT_TIMEOUT, enable_gloo: bool = False
):
    torch.cuda.set_device(get_world().local_rank)
    backend = None  # nccl by default
    if enable_gloo:
        get_logger().info("Using gloo backend for torch.distributed")
        backend = "cpu:gloo,cuda:nccl"
    dist.init_process_group(backend=backend, timeout=timeout)


def print0(*args, **kwargs):
    if get_world().rank == 0:
        print(*args, **kwargs)


def log0(*args, **kwargs):
    if get_world().rank == 0:
        get_logger().info(*args, **kwargs)
