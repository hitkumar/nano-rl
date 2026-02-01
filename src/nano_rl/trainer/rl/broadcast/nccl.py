import pickle
import time
from pathlib import Path
from typing import cast, Iterator

import torch
import torch.nn as nn
from nano_rl.trainer.rl.broadcast.base import WeightBroadcast
from nano_rl.trainer.rl.config import NCCLWeightBroadcastConfig
from nano_rl.trainer.world import get_world
from nano_rl.utils.pathing import get_broadcasts_dir, get_step_path
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

# Marker file that inference workers create to signal they're ready for NCCL broadcast.
NCCL_READY_MARKER = "NCCL_READY"


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """
    nccl broadcast only works with tensors, so we wrap the integer in a tensor before broadcast
    """
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    # source 0 is trainer
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(
    state_dict: dict[str, Tensor], communicator: PyNcclCommunicator
) -> None:
    # Group state dict tensors by dtype so that we can concatenate tensors of same dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), "DTensor is not supported for broadcast"
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata that tells receivers how to construct the state dict
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata to all receivers
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    # list converts bytes to a list of integers which can be converted to a tensor
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    # Concatenate all tensors for a given dtype and broadcast
    for dtype, items in dtype_groups.items():
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors).cuda()
        communicator.broadcast(concatenated, src=0)
        # Free up GPU memory
        del concatenated
        for _, value in items:
            del value


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum layer number from a state dict.

    HuggingFace models have keys like "model.layers.0.self_attn.q_proj.weight".
    This finds the highest layer index to know how many layers to iterate over.
    """
    max_layer = -1
    for key in state_dict.keys():
        if "model.layers." in key:
            # Split key like "model.layers.0.self_attn.q_proj.weight" into parts.
            parts = key.split(".")
            # Find the index of "layers" and the next part is the layer number.
            layer_idx = parts.index("layers") + 1
            if layer_idx < len(parts):
                try:
                    layer_num = int(parts[layer_idx])
                    max_layer = max(max_layer, layer_num)
                except ValueError:
                    # Not a number, skip (shouldn't happen in normal models).
                    pass
    return max_layer


def filter_state_dict_by_layers(
    state_dict: dict[str, Tensor], num_layers: int
) -> Iterator[tuple[int, dict[str, Tensor]]]:
    """
    Yields an iterator of state_dict for each layer as well as the remaining weights.
    Yields:
        (layer_id, layer_state_dict) tuples where:
        - layer_id 0 = non-layer weights (embeddings, final norm, lm_head)
        - layer_id 1..N+1 = individual transformer layers
    """
    yield 0, {
        key: value for key, value in state_dict.items() if "model.layers." not in key
    }
    for i in range(num_layers + 1):
        yield (
            i + 1,  # Offset by 1 since we used 0 for non-layer weights above.
            {
                key: value
                for key, value in state_dict.items()
                # Match keys starting with "model.layers.{i}." to get all params for layer i.
                if key.startswith(f"model.layers.{i}.")
            },
        )


class NCCLWeightBroadcast(WeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.world = get_world()
        self.dtype = dtype

        # Same directory structure as filesystem broadcast for orchestrator compatibility.
        self.broadcasts_dir = get_broadcasts_dir(output_dir)
        self.communicator: PyNcclCommunicator | None = None

        # Only master rank (rank 0) participates in NCCL broadcast
        if self.world.is_master:
            pg = StatelessProcessGroup.create(
                host=config.host,
                port=config.port,
                rank=0,
                world_size=config.inference_world_size + 1,
                store_timeout=config.timeout,  # timeout for initial connection
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.info(
                f"NCCL broadcast initialized (world_size={config.inference_world_size + 1})"
            )
        else:
            # Non-master ranks don't need a communicator.
            # They still call broadcast_weights() but skip the actual NCCL calls.
            self.logger.debug(
                "NCCL broadcast initialized on non-master rank (no communicator)"
            )

    def _notify_orchestrator(self, step: int) -> Path:
        """notify orch that weights are ready for NCCL broadcast"""
        save_dir = get_step_path(self.broadcasts_dir, step)
        save_dir.mkdir(exist_ok=True, parents=True)
        stable_file = save_dir / "STABLE"
        stable_file.touch()
        self.logger.debug(f"notified orch at {stable_file}")
        return save_dir

    def _wait_for_nccl_ready(
        self, save_dir: Path, interval: float = 0.01, timeout: float = 300
    ) -> None:
        """
        Wait for inference servers to signal they are ready to receive NCCL broadcast
        The flow is:
        1. Trainer touches STABLE
        2. Orchestrator sees STABLE, calls /update_weights on inference servers
        3. Inference workers init their NCCL receivers and touch NCCL_READY
        4. Trainer sees NCCL_READY and starts broadcasting

        This synchronization is necessary because NCCL operations will hang if
        the receiver isn't ready when the sender starts broadcasting.
        """
        nccl_ready_file = save_dir / NCCL_READY_MARKER
        self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")

        start = time.perf_counter()
        while not nccl_ready_file.exists():
            if time.perf_counter() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for NCCL_READY at {nccl_ready_file}"
                )
            time.sleep(interval)

        self.logger.debug("Inference workers ready for NCCL broadcast")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """
        Broadcast model state dict into the inference pool using NCCL
        Args:
            model: The model to broadcast (FSDP-wrapped in distributed training).
            step: Current training step, used for the notification directory.
        """
        start_time = time.perf_counter()
        if self.world.is_master:
            save_dir = self._notify_orchestrator(step)
            self._wait_for_nccl_ready(save_dir)

        # sharded tensors per GPU
        state_dict = model.state_dict()
        num_layers = get_max_layer_num(state_dict)
        # get_max_layer_num returns 0 indexed number of layers in the model (eg. if model has 32 layers it will return 31), this is why we need +2 here.
        num_state_dict_to_send = num_layers + 2

        # Tell receivers how many state_dicts to expect so they know how many times to loop over
        if self.world.is_master and self.communicator is not None:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        for layer_id, layer_state_dict in filter_state_dict_by_layers(
            state_dict, num_layers
        ):
            for key, value in list(layer_state_dict.items()):
                if isinstance(value, DTensor):
                    value = cast(DTensor, value.to(self.dtype)).full_tensor()
                layer_state_dict[key] = value

            if self.world.is_master and self.communicator is not None:
                broadcast_state_dict(layer_state_dict, self.communicator)

        if self.world.is_master:
            self.logger.info(
                f"Weights broadcasted via NCCL in {time.perf_counter() - start_time:.2f}s"
            )
