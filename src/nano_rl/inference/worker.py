import pickle
from typing import cast, Iterator, TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# we use the actual worker class for type checking, but in runtime object is used as VLLM injects the desired attributes at runtime.
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([0], dtype=torch.long).to(communicator.device)
    # rank 0 sends and all other ranks receive
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(
    communicator: PyNcclCommunicator,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Stream tensors in a state dict broadcasted over nccl
    This is the receiver counterpart to broadcast_state_dict() in nccl.py.
    """
    # receive the size of pickled metadata, trainer sends this and inference workers receive
    size_tensor = torch.tensor([0], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)

    # receive metadata
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(
        communicator.device
    )
    communicator.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    # For each dtype group, receive the concatenated tensor and split into individual tensors
    for dtype, tensor_info_list in metadata.items():
        total_elements = sum(numel for _, _, numel in tensor_info_list)

        # receive the concatenated tensor
        concatenated = torch.empty(
            total_elements, dtype=dtype, device=communicator.device
        )
        communicator.broadcast(concatenated, src=0)
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        # Free the concatented buffer after all tensors are extracted.
        del concatenated


class NCCLWeightBroadcastReceiver:
    """Receives weight state dicts from trainer via NCCL broadcasts
    Inference counterpart to NCCLWeightBroadcast in the trainer"""

    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
    ):
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            store_timeout=timeout,
        )
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Receives the state dict of a model from trainer rank via NCCL."""
        num_state_dict_to_receive = receive_integer(self.communicator)

        for layer_id in range(num_state_dict_to_receive):
            for key, value in receive_state_dict(self.communicator):
                yield key, value


class WeightUpdateWorker(Worker):
    """vllm worker extension for updating model weights
    Supports two modes:
    - Filesystem: Load weights from a directory path (default)
    - NCCL: Receive weights directly from trainer via GPU-to-GPU broadcast

    The mode is determined by whether init_broadcaster() was called.
    If init_broadcaster() was called, update_weights() uses NCCL.
    Otherwise, it loads from the filesystem path.

    Note: We don't define __init__ because vLLM injects attributes dynamically.
    Defining __init__ would break vLLM's worker extension mechanism.
    """

    def init_broadcaster(
        self,
        host: str,
        port: int,
        server_rank: int,
        num_inference_server: int,
        timeout: int,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Called via collective_rpc from the /init_broadcaster endpoint.
        All GPU workers in the engine call this method with the same args.

        Args:
            host: TCP store host for process group rendezvous (same as trainer config)
            port: TCP store port for process group rendezvous (same as trainer config)
            server_rank: This inference server's index (0, 1, 2, ...).
                        Passed from orchestrator which tracks server order.
            num_inference_server: Total number of inference servers.
            timeout: Timeout in seconds for process group initialization.
        """
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank
        # Compute this GPU's global rank across all inference workers.
        # Example with 2 servers, TP=2:
        #   Server 0: GPU0 -> global_rank 0, GPU1 -> global_rank 1
        #   Server 1: GPU0 -> global_rank 2, GPU1 -> global_rank 3
        global_rank_inference = (server_rank * tp_size) + tp_rank
        global_inference_world_size = num_inference_server * tp_size

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,
            world_size=global_inference_world_size + 1,
            device=self.device,
            timeout=timeout,
        )

    def update_weights(self, weight_path: str) -> None:
        model = getattr(self.model_runner.model, "runnable", self.model_runner.model)
        assert isinstance(model, Module)

        # Check if init_broadcaster was called
        if hasattr(self, "nccl_broadcast_receiver"):
            state_iter = self.nccl_broadcast_receiver.receive_state_dict()
            model.load_weights(state_iter)
        else:
            # File system mode
            model_loader = get_model_loader(self.load_config)
            assert isinstance(model_loader, DefaultModelLoader)

            source = DefaultModelLoader.Source(
                weight_path,
                revision=None,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(
                    model, "allow_patterns_overrides", None
                ),
            )
            model.load_weights(model_loader._get_weights_iterator(source))

        # Apply post processing like quantization
        process_weights_after_loading(
            model, self.model_runner.model_config, next(model.parameters()).device
        )
