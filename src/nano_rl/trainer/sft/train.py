"""
SFT trainer
"""

import time
from datetime import timedelta

import torch
import torch.distributed as dist
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from nano_rl.trainer.ckpt import WeightCheckpointManager
from nano_rl.trainer.model import setup_model, setup_tokenizer
from nano_rl.trainer.optim import setup_optimizer

from nano_rl.trainer.parallel_dims import get_parallel_dims
from nano_rl.trainer.scheduler import setup_scheduler
from nano_rl.trainer.sft.config import SFTTrainerConfig
from nano_rl.trainer.sft.data import setup_dataloader, setup_dataset
from nano_rl.trainer.utils import print0, setup_torch_distributed
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import setup_logger
from nano_rl.utils.pydantic_config import parse_argv
from torch.nn import CrossEntropyLoss


def train(config: SFTTrainerConfig):
    # Setup distributed training
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=(
            config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log"
            if config.log.file
            else None
        ),
    )
    # gloo is only needed for cpu offload
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload,
    )
    # This will only work correctly if placed after setup_torch_distributed.
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    weight_manager = WeightCheckpointManager(config.output_dir, config=config.ckpt)

    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)
    # batch_size is the unique number of samples we see before one optimizer step
    # since non_data_parallel_size GPUs process the same batch, we need more micro batches if cp/tp is enabled.
    total_micro_batches = config.data.batch_size * parallel_dims.non_data_parallel_size
    samples_per_fwd_pass = config.data.micro_batch_size * world.world_size
    assert (
        total_micro_batches % samples_per_fwd_pass == 0
    ), f"total_micro_batches ({total_micro_batches}) % samples_per_fwd_pass ({samples_per_fwd_pass}) != 0"
    grad_accum_steps = total_micro_batches // samples_per_fwd_pass
    # logger.info(f"grad_accum_steps: {grad_accum_steps}")
    assert (
        grad_accum_steps > 0
    ), f"grad_accum_steps ({grad_accum_steps}) must be greater than 0"

    # Model & tokenizer

    print0(f"Loading model: {config.model.name}")
    model = setup_model(config.model, parallel_dims)

    print0(f"Loading tokenizer: {config.tokenizer.name}")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Optimizer & scheduler
    optimizer = setup_optimizer(
        config.optim, model, parallel_dims.world_mesh["dp_shard_cp"]
    )
    scheduler = setup_scheduler(
        optimizer, config.scheduler, config.max_steps, config.optim.lr
    )

    # Data
    print0(f"Loading dataset")
    dataset = setup_dataset(
        tokenizer, config.data, parallel_dims.non_data_parallel_size
    )
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # loss function
    match config.loss_impl:
        case "liger":
            ce_loss = LigerCrossEntropyLoss(reduction="none")
        case "torch":
            ce_loss = CrossEntropyLoss(reduction="none")
        case _:
            raise ValueError(f"Unknown loss implementation: {config.loss_impl}")

    print0(f"Starting training (max_steps={config.max_steps})")
    step = 0

    while step < config.max_steps:
        start_time = time.perf_counter()
        batch_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        for _ in range(grad_accum_steps):
            batch = next(dataiter)
            input_ids = batch["input_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # forward pass
            logits = model(input_ids=input_ids, position_ids=position_ids).logits
            B, L, V = logits.shape
            # logger.info(
            #     f"Batch input_ids shape: {input_ids.shape}, logits shape: {logits.shape}"
            # )

            # Loss with masking
            loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)
            loss = loss[loss_mask].mean()

            # Delete logits before backward pass to avoid memory spike
            del logits

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            # just accumulate the loss for logging.
            batch_loss += scaled_loss.detach()

        # Does gradient clipping in place
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm
        ).full_tensor()

        # total_norm = sqrt(
        #     sum(x.grad**2 for x in model.parameters() if x.grad is not None)
        # )
        optimizer.step()
        scheduler.step()

        # synchronize loss across all ranks, only for logging and monitoring
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

        step_time = time.perf_counter() - start_time
        print0(
            f"rank: {world.rank} | step {step} | Loss {batch_loss.item():.4f} | Grad Norm {grad_norm:.4f} | Step Time {step_time:.2}s"
        )
        if config.ckpt.interval and step > 0 and step % config.ckpt.interval == 0:
            weight_manager.save(step, model, tokenizer)
        step += 1

    # save final checkpoint
    weight_manager.save(step, model, tokenizer)
    print0("Training done")


def main():
    config = parse_argv(SFTTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
