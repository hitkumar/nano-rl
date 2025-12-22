from nano_rl.trainer.config import SchedulerConfigType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler


def setup_constant_scheduler(optimizer: Optimizer) -> LRScheduler:
    return ConstantLR(optimizer, factor=1.0)


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfigType,
    max_steps: int | None,
    lr: float,
) -> LRScheduler:
    match scheduler_config.type:
        case "constant":
            return setup_constant_scheduler(optimizer)
        case _:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config.type}")
