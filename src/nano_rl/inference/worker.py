from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# we use the actual worker class for type checking, but in runtime object is used as VLLM injects the desired attributes at runtime.
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class WeightUpdateWorker(Worker):
    def init_broadcaster(self, *args, **kwargs) -> None:
        pass

    def update_weights(self, weight_path: str) -> None:
        model = self.model_runner.model.runnable
        assert isinstance(model, Module)

        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)

        source = DefaultModelLoader.Source(
            weight_path,
            revision=None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patters_override=getattr(model, "allow_patterns_overrides", None),
        )
        model.load_weights(model_loader._get_weights_iterator(source))

        # Apply post processing like quantization
        process_weights_after_loading(
            model, self.model_runner.model_config, next(model.parameters()).device
        )
