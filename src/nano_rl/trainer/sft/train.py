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
from nano_rl.trainer.perf import get_perf_counter
from nano_rl.trainer.scheduler import setup_scheduler
from nano_rl.trainer.sft.config import SFTTrainerConfig
from nano_rl.trainer.sft.data import setup_dataloader, setup_dataset
from nano_rl.trainer.utils import log0, print_benchmark, setup_torch_distributed
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import setup_logger
from nano_rl.utils.monitor import setup_monitor
from nano_rl.utils.pydantic_config import parse_argv
from nano_rl.utils.utils import to_col_format
from torch.nn import CrossEntropyLoss


def train(config: SFTTrainerConfig) -> None:
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

    # metrics monitor
    monitor = setup_monitor(output_dir=config.output_dir)

    weight_manager = WeightCheckpointManager(config.output_dir, config=config.ckpt)

    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)
    log0(f"Parallel dimensions: {parallel_dims}")
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

    log0(f"Loading model: {config.model.name}")
    model = setup_model(config.model, parallel_dims)

    log0(f"Loading tokenizer: {config.tokenizer.name}")
    tokenizer = setup_tokenizer(config.tokenizer)

    log0(f"grad accum steps is {grad_accum_steps}")

    # Optimizer & scheduler
    optimizer = setup_optimizer(
        config.optim, model, parallel_dims.world_mesh["dp_shard_cp"]
    )
    scheduler = setup_scheduler(
        optimizer, config.scheduler, config.max_steps, config.optim.lr
    )

    # Data
    log0(f"Loading dataset")
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

    log0(f"Starting training (max_steps={config.max_steps})")
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

        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # synchronize loss across all ranks, only for logging and monitoring
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

        # Benchmark training
        seq_len = config.data.seq_len
        # batch_size is global (unique samples per step), so total tokens = batch_size * seq_len
        num_tokens = config.data.batch_size * seq_len
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_sec() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # convert to GiB

        step_time = time.perf_counter() - start_time
        monitor.log(
            {
                "loss/mean": batch_loss.mean(),
                # Optimizer metrics
                "optim/grad_norm": grad_norm.item(),
                "optim/lr": current_lr,
                # performance metrics
                "perf/throughput": throughput,
                "perf/mfu": mfu,
                "perf/peak_memory": peak_memory,
                # Time metrics
                "time/step": step_time,
            },
            step=step,
        )

        if config.ckpt.interval and step > 0 and step % config.ckpt.interval == 0:
            weight_manager.save(step, model, tokenizer)
        step += 1

    # save final checkpoint
    weight_manager.save(step, model, tokenizer)
    if world.is_master:
        history = to_col_format(monitor.history)
        print_benchmark(history)
        monitor.close()

    log0("Training done")


def main():
    config = parse_argv(SFTTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
