"""RL training loop"""

import time
from datetime import timedelta

import torch
import torch.distributed as dist
from nano_rl.trainer.ckpt import WeightCheckpointManager
from nano_rl.trainer.model import setup_model, setup_tokenizer
from nano_rl.trainer.optim import setup_optimizer
from nano_rl.trainer.parallel_dims import get_parallel_dims
from nano_rl.trainer.rl.config import RlTrainerConfig
from nano_rl.trainer.rl.data import DataLoader, FakeDataLoader
from nano_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    shift_logits,
)
from nano_rl.trainer.scheduler import setup_scheduler
from nano_rl.trainer.utils import log0, setup_torch_distributed
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import setup_logger
from nano_rl.utils.pydantic_config import parse_argv


def train(config: RlTrainerConfig) -> None:
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=(
            config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log"
            if config.log.file
            else None
        ),
    )
    logger.info(f"Starting RL training on rank {world.rank}...")
    # gloo is only needed for cpu offload
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload,
    )
    # This will only work correctly if placed after setup_torch_distributed.
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    weight_manager = WeightCheckpointManager(config.output_dir, config=config.ckpt)

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    log0(f"Loading model {config.model.name}")
    model = setup_model(config.model, parallel_dims)

    log0(f"Loading tokenizer: {config.tokenizer.name}")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Optimizer & scheduler
    optimizer = setup_optimizer(
        config.optim, model, parallel_dims.world_mesh["dp_shard_cp"]
    )
    scheduler = setup_scheduler(
        optimizer, config.scheduler, config.max_steps, config.optim.lr
    )
    log0("setting up data loader")
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake, config.model.seq_len)
    else:
        dataloader = DataLoader(
            output_dir=config.output_dir,
            start_step=0,
            seq_len=config.model.seq_len,
            tokenizer=tokenizer,
            config=config.rollout_transport,
        )

    log0("starting training loop")
    step = 0
    max_steps = config.max_steps or float("inf")
    while step < max_steps:
        step_start_time = time.perf_counter()

        # block until orchestrator writes a batch to filesystem
        dataloader.wait_for_batch()
        batch = dataloader.get_batch()

        # move batch to device
        input_ids = batch["input_ids"].to(device)
        position_ids = batch["position_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        advantages = batch["advantages"].to(device)
        inference_logprobs = batch["inference_logprobs"].to(device)
        temperature = batch["temperature"]

        optimizer.zero_grad()
        # forward pass
        logits = model(input_ids=input_ids, position_ids=position_ids).logits
        logits = logits.float().contiguous()

        # we shift here to match inference logprobs
        shifted_logits = shift_logits(logits)
        shifted_logits = shifted_logits / temperature
        trainer_logprobs = selective_log_softmax(shifted_logits, input_ids)

        loss, loss_diagnostics = compute_loss(
            trainer_logprobs=trainer_logprobs,
            inference_logprobs=inference_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            loss_config=config.loss,
        )

        # compute entropy for monitoring, lower is better
        entropy = compute_entropy(shifted_logits)
        mean_entropy = entropy[loss_mask].mean()

        del logits, shifted_logits

        # backward pass
        loss.backward()
        # grad clipping, full_tensor returns the full tensor from all FSDP shards
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm
        ).full_tensor()

        # update weights
        optimizer.step()
        scheduler.step()

        # synchronize loss across all ranks, only for logging and monitoring
        # TODO: currently a no-op as all shards operate on the same batch, fix this.
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        step_time = time.perf_counter() - step_start_time

        log0(
            f"Step {step} | "
            f"Loss {loss.item():.4f} | "
            f"Entropy {mean_entropy.item():.4f} | "
            f"KL {loss_diagnostics['mismatch_kl'].item():.4f} | "
            f"Masked {loss_diagnostics['tokens_masked'].item():.2%} | "
            f"Grad Norm {grad_norm:.4f} | "
            f"Time {step_time:.2f}s"
        )

        # save weights at regular intervals
        if config.ckpt.interval and step % config.ckpt.interval == 0:
            weight_manager.save(step, model, tokenizer)

        step += 1

    # save final checkpoint
    weight_manager.save(step, model, tokenizer)
    log0("Training done")


def main():
    config = parse_argv(RlTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
