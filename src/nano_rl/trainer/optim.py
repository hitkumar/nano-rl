from multiprocessing.sharedctypes import Value

from nano_rl.trainer.config import OptimizerConfigType
from torch import nn
from torch.optim import AdamW, Optimizer, SGD


def setup_optimizer(config: OptimizerConfigType, model: nn.Module) -> Optimizer:
    match config.type:
        case "sgd":
            return SGD(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        case _:
            raise ValueError(f"Unsupported optimizer type: {config.type}")
