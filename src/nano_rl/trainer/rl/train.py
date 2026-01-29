"""RL training loop"""

import time
from datetime import timedelta

import torch
import torch.distributed as dist
from nano_rl.trainer.ckpt import WeightCheckpointManager
from nano_rl.trainer.model import setup_model, setup_tokenizer
from nano_rl.trainer.optim import setup_optimizer
from nano_rl.trainer.parallel_dims import get_parallel_dims
from nano_rl.trainer.perf import get_perf_counter
from nano_rl.trainer.rl.broadcast import setup_weight_broadcast
from nano_rl.trainer.rl.config import RlTrainerConfig
from nano_rl.trainer.rl.data import DataLoader, FakeDataLoader, TensorBatch
from nano_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    shift_logits,
)
from nano_rl.trainer.scheduler import setup_scheduler
from nano_rl.trainer.utils import log0, print_benchmark, setup_torch_distributed
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import setup_logger
from nano_rl.utils.monitor import setup_monitor
from nano_rl.utils.pydantic_config import parse_argv
from nano_rl.utils.utils import to_col_format


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

    # metrics monitor
    monitor = setup_monitor(output_dir=config.output_dir)

    weight_manager = WeightCheckpointManager(config.output_dir, config=config.ckpt)

    # broadcasts weights for inference server.
    weight_broadcaster = setup_weight_broadcast(
        config.output_dir, config.weight_broadcast, config.max_async_level
    )

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    log0(f"Loading model {config.model.name}")
    log0(f"Parallel dimensions: {parallel_dims}")
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
        dataloader = FakeDataLoader(
            config.data.fake, config.model.seq_len, parallel_dims.dp_degree
        )
    else:
        dataloader = DataLoader(
            output_dir=config.output_dir,
            start_step=0,
            seq_len=config.model.seq_len,
            tokenizer=tokenizer,
            config=config.rollout_transport,
            dp_world_size=parallel_dims.dp_degree,
        )

    log0("starting training loop")
    log0(f"Config seq_len: {config.model.seq_len}")

    # Initialize perf counter once with config seq_len
    perf_counter = get_perf_counter(model, config.model.seq_len)
    log0(f"FLOPs per token: {perf_counter.num_flop_per_token:.2e}")
    log0(f"GPU peak FLOPs: {perf_counter.gpu_peak_flops:.2e}")

    step = 0
    max_steps = config.max_steps or float("inf")
    while step < max_steps:
        wait_start = time.perf_counter()

        # block until orchestrator writes a batch to filesystem
        dataloader.wait_for_batch()
        batch: TensorBatch = dataloader.get_batch()
        wait_time = time.perf_counter() - wait_start

        compute_start = time.perf_counter()
        # Accumulate gradients over micro-batches
        optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0
        total_entropy = 0.0
        total_off_policy_step = 0
        total_mismatch_kl = 0.0
        total_tokens_masked = 0.0

        # Split batch into chunks for gradient accumulation
        grad_accum_steps = config.gradient_accumulation_steps
        batch_size = batch["input_ids"].shape[0]
        assert batch_size % grad_accum_steps == 0
        chunk_size = batch_size // grad_accum_steps

        for i in range(grad_accum_steps):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < grad_accum_steps - 1 else batch_size

            off_policy_step = step - batch["ckpt_step"]
            total_off_policy_step += off_policy_step

            # move batch chunk to device
            input_ids = batch["input_ids"][start_idx:end_idx].to(device)
            position_ids = batch["position_ids"][start_idx:end_idx].to(device)
            loss_mask = batch["loss_mask"][start_idx:end_idx].to(device)
            advantages = batch["advantages"][start_idx:end_idx].to(device)
            inference_logprobs = batch["inference_logprobs"][start_idx:end_idx].to(
                device
            )
            temperature = batch["temperature"]

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
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            total_loss += loss.item()
            total_tokens += input_ids.numel()
            total_mismatch_kl += loss_diagnostics["mismatch_kl"].item()
            total_tokens_masked += loss_diagnostics["tokens_masked"].item()

            # compute entropy for monitoring, lower is better
            entropy = compute_entropy(shifted_logits)
            mean_entropy = entropy[loss_mask].mean()
            total_entropy += mean_entropy.item()

            del logits, shifted_logits, trainer_logprobs, loss, scaled_loss
            # torch.cuda.empty_cache()

        # Average loss across gradient accumulation steps
        avg_loss = torch.tensor(total_loss / grad_accum_steps, device=device)
        avg_entropy = torch.tensor(total_entropy / grad_accum_steps, device=device)
        avg_off_policy_step = total_off_policy_step / grad_accum_steps
        avg_mismatch_kl = total_mismatch_kl / grad_accum_steps
        avg_tokens_masked = total_tokens_masked / grad_accum_steps
        # synchronize loss across all ranks, only for logging and monitoring
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_entropy, op=dist.ReduceOp.AVG)

        # grad clipping, full_tensor returns the full tensor from all FSDP shards
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm
        ).full_tensor()

        # update weights
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        compute_time = time.perf_counter() - compute_start

        # total_tokens is tokens processed by this rank only
        # multiply by dp_degree to get global tokens for MFU calculation
        global_tokens = total_tokens * parallel_dims.dp_degree

        # Log batch dimensions on first step for debugging
        if step == 0:
            log0(
                f"Batch dims: rank={world.rank} batch_size={batch_size}, seq_len={config.model.seq_len}, local_tokens={total_tokens}, global_tokens={global_tokens}"
            )

        perf_counter.count_tokens(global_tokens)
        throughput = perf_counter.get_tokens_per_sec() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # convert to GiB

        step_time = wait_time + compute_time

        log0(
            f"Step {step} | "
            f"Loss {avg_loss.item():.4f} | "
            f"Wait {wait_time:.2f}s | "
            f"Compute {compute_time:.2f}s | "
            f"MFU {mfu:.2f}% |"
            f"batch_size {batch_size}|"
            f"Throughput {throughput:.0f} tok/s"
        )

        monitor.log(
            {
                "loss/mean": avg_loss.item(),
                "loss/entropy": avg_entropy.item(),
                "loss/mismatch_kl": avg_mismatch_kl,
                "loss/tokens_masked": avg_tokens_masked,
                # Optimizer metrics
                "optim/grad_norm": grad_norm.item(),
                "optim/lr": current_lr,
                # Training metrics
                "train/off_policy_step": avg_off_policy_step,
                # performance metrics
                "perf/throughput": throughput,
                "perf/mfu": mfu,
                "perf/peak_memory": peak_memory,
                # Time metrics
                "time/step": step_time,
                "time/wait": wait_time,
                "time/compute": compute_time,
            },
            step=step,
        )

        # broadcast weights to inference server (every broadcast_interval steps)
        broadcast_time = 0.0
        if step > 0 and step % config.broadcast_interval == 0:
            broadcast_start = time.perf_counter()
            weight_broadcaster.broadcast_weights(model, step)
            broadcast_time = time.perf_counter() - broadcast_start
        monitor.log({"time/broadcast": broadcast_time}, step=step)

        # save weights at regular intervals
        if step > 0 and config.ckpt.interval and step % config.ckpt.interval == 0:
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
    config = parse_argv(RlTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
