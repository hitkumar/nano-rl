"""
Coordinates rollout generation with weight updates in an async RL training loop.
"""

import asyncio
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

import verifiers as vf
from httpx import AsyncClient
from nano_rl.orchestrator.config import OrchestratorConfig
from nano_rl.orchestrator.utils import get_sampling_args
from nano_rl.transport import TrainingBatchSender  # Trainer communication
from nano_rl.utils.client import update_weights  # Weight updates
from nano_rl.utils.logger import get_logger
from nano_rl.utils.pathing import (
    get_broadcasts_dir,  # output_dir / "broadcasts"
    get_step_path,  # weights_dir / "step_{n}"
    resolve_latest_ckpt_dir,  # Find newest ckpt number
)
from nano_rl.utils.vf import generate_group, get_seq_len  # Rollout generation
from openai import AsyncOpenAI


@dataclass
class SchedulerStats:
    """Tracks scheduler performance metrics for throughput analysis."""

    # Rollout generation stats
    total_rollouts: int = 0
    total_tokens_generated: int = 0
    total_inference_time: float = 0.0  # sum of per-call times (for per-client stats)
    total_batch_time: float = 0.0  # wall-clock time for batches (for throughput)
    total_batches: int = 0

    # Checkpoint wait stats
    total_checkpoint_wait_time: float = 0.0
    checkpoint_wait_count: int = 0

    # Per-client stats (keyed by base_url)
    client_rollouts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    client_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    client_time: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def record_rollout(
        self, client_url: str, num_rollouts: int, tokens: int, elapsed: float
    ) -> None:
        self.total_rollouts += num_rollouts
        self.total_tokens_generated += tokens
        self.total_inference_time += elapsed
        self.client_rollouts[client_url] += num_rollouts
        self.client_tokens[client_url] += tokens
        self.client_time[client_url] += elapsed

    def record_checkpoint_wait(self, wait_time: float) -> None:
        self.total_checkpoint_wait_time += wait_time
        self.checkpoint_wait_count += 1

    def record_batch(self, batch_time: float, batch_tokens: int) -> None:
        self.total_batches += 1
        self.total_batch_time += batch_time

    @property
    def tokens_per_second(self) -> float:
        """Wall-clock throughput (tokens per second of actual time)."""
        if self.total_batch_time == 0:
            return 0.0
        return self.total_tokens_generated / self.total_batch_time

    @property
    def avg_checkpoint_wait(self) -> float:
        if self.checkpoint_wait_count == 0:
            return 0.0
        return self.total_checkpoint_wait_time / self.checkpoint_wait_count


class Scheduler:
    def __init__(
        self,
        config: OrchestratorConfig,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        envs: list[vf.Environment],
        # unused for now, we directly get the examples from the envs
        # examples_iter,  # iterator yielding training examples (prompts)
        sender: TrainingBatchSender,  # sends completed rollout batches to trainer.
    ):
        self.config = config
        self.clients = clients
        self.client_cycle = cycle(clients)  # round-robin iterator
        # used to update weights, do health checks.
        self.admin_clients = admin_clients
        self.envs = envs
        # self.examples_iter = examples_iter
        self.sender = sender
        # assumes that logger has been initialized elsewhere
        self.logger = get_logger()

        # used by verifiers
        self.sampling_args = get_sampling_args(config.sampling)

        # Orchestrator reads from broadcasts directory to know about new checkpoints
        self.broadcasts_dir = get_broadcasts_dir(config.output_dir)

        # Keep track of current state
        self.current_weight_step = (
            -1
        )  # checkpoint inference server has loaded to generate rollouts, -1 means using base model.
        self.step = 0  # training step we are generating rollouts for
        self._stop = False  # shutdown signal

        # async level control; blocks batch generation when inference is too far ahead
        # Like a boolean flag that coroutines can wait on using event.wait()
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()

        self.logger.info(
            f"Scheduler initialized with {len(clients)} inference client(s)"
        )

        # Performance tracking
        self.stats = SchedulerStats()

    @property
    def async_level(self) -> int:
        """
        step = 5              → "We're generating rollouts FOR training step 5"
        current_weight_step = 3  → "We're USING weights from checkpoint 3"
        async_level = 2       → "Orchestrator is 2 steps AHEAD of trainer"
        """
        return self.step - max(self.current_weight_step, 0)

    async def _update_policy(self) -> None:
        """Checks for new broadcast weights and updates if ready"""

        latest_step = resolve_latest_ckpt_dir(self.broadcasts_dir)
        if latest_step is not None and latest_step > self.current_weight_step:
            weights_path = get_step_path(self.broadcasts_dir, latest_step)
            update_start = time.perf_counter()
            await update_weights(self.admin_clients, weights_path)
            update_time = time.perf_counter() - update_start
            self.current_weight_step = latest_step
            self.logger.info(
                f"Updated to weights step: {latest_step} (took {update_time:.2f}s)"
            )
            # possibly unblock batch generation
            self.checkpoint_ready.set()

    async def update_policy_loop(self) -> None:
        """Poll for new weights and update inference server"""
        while not self._stop:
            await self._update_policy()
            await asyncio.sleep(0.5)  # Poll every 500ms

    async def _wait_for_checkpoint(self, required_step: int) -> float:
        """Wait until a checkpoint at or after required_step is available.

        Returns the time spent waiting.
        """
        wait_start = time.perf_counter()
        while self.current_weight_step < required_step:
            self.logger.info(
                f"Waiting for ckpt to become available, currently at {self.current_weight_step}, need {required_step}"
            )
            self.checkpoint_ready.clear()  # set this to false
            await self.checkpoint_ready.wait()
        return time.perf_counter() - wait_start

    def _get_next_client(self) -> AsyncOpenAI:
        """Get next client from round-robin cycle"""
        return next(self.client_cycle)

    async def schedule_group_rollout(
        self, env: vf.Environment, example: dict[str, Any], client: AsyncOpenAI
    ) -> list[vf.State]:
        client_url = str(client.base_url)
        start = time.perf_counter()
        states = await generate_group(
            client=client,
            env=env,
            model_name=self.config.model.name,
            example=example,
            rollouts_per_example=self.config.rollouts_per_example,
            sampling_args=self.sampling_args,
        )
        elapsed = time.perf_counter() - start

        # Count tokens generated (prompt + completion to match trainer's token counting)
        total_tokens = sum(get_seq_len(state) for state in states)

        self.stats.record_rollout(
            client_url=client_url,
            num_rollouts=len(states),
            tokens=total_tokens,
            elapsed=elapsed,
        )

        self.logger.debug(
            f"group_rollout took {elapsed:.2f}s for {len(states)} rollouts, "
            f"{total_tokens} tokens on {client_url}"
        )
        return states

    async def generate_batch(self, step: int) -> list[vf.State]:
        """Generates a batch of rollouts for training step"""
        self.step = step
        batch_start = time.perf_counter()
        wait_time = 0.0

        if self.config.max_async_level > 0:  # 0 means disabled
            min_required_step = self.step - self.config.max_async_level
            if self.current_weight_step < min_required_step:
                wait_time = await self._wait_for_checkpoint(min_required_step)
                self.stats.record_checkpoint_wait(wait_time)

        examples_needed = self.config.batch_size // self.config.rollouts_per_example
        tasks = []

        task_creation_start = time.perf_counter()
        for i in range(examples_needed):
            env = self.envs[i % len(self.envs)]  # round robin across environments
            client = self._get_next_client()  # round robin across inference servers
            dataset = env.get_dataset()
            example = random.choice(dataset)
            task = asyncio.create_task(
                self.schedule_group_rollout(env, example, client)
            )
            tasks.append(task)
        task_creation_time = time.perf_counter() - task_creation_start

        # wait for all tasks to complete (they run in parallel)
        gather_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        gather_time = time.perf_counter() - gather_start

        all_states = []
        for result in results:
            all_states.extend(result)

        # Count tokens in this batch for throughput calculation
        batch_tokens = sum(get_seq_len(state) for state in all_states)
        rollout_time = time.perf_counter() - batch_start - wait_time  # exclude checkpoint wait
        self.stats.record_batch(batch_time=rollout_time, batch_tokens=batch_tokens)

        total_time = time.perf_counter() - batch_start
        self.logger.info(
            f"generate_batch timing: total={total_time:.2f}s, "
            f"wait_ckpt={wait_time:.2f}s, task_creation={task_creation_time:.3f}s, "
            f"rollout_gather={gather_time:.2f}s, examples={examples_needed}, "
            f"throughput={self.stats.tokens_per_second:.1f} tok/s"
        )

        return all_states

    def stop(self):
        self._stop = True

    def get_metrics(self) -> dict[str, float]:
        """Return scheduler metrics for logging."""
        metrics = {
            "scheduler/async_level": self.async_level,
            "scheduler/weight_step": self.current_weight_step,
            "scheduler/train_step": self.step,
            # Throughput metrics (inference side)
            "perf/inference_throughput": self.stats.tokens_per_second,
            "perf/inference_tokens": self.stats.total_tokens_generated,
            "perf/inference_rollouts": self.stats.total_rollouts,
            "perf/inference_batches": self.stats.total_batches,
            # Wait time metrics (indicates trainer bottleneck)
            "time/checkpoint_wait_total": self.stats.total_checkpoint_wait_time,
            "time/checkpoint_wait_avg": self.stats.avg_checkpoint_wait,
            "time/checkpoint_wait_count": self.stats.checkpoint_wait_count,
        }

        # Per-client metrics (to check load balancing across inference servers)
        for client_url, rollouts in self.stats.client_rollouts.items():
            # Use a short key based on port or index
            short_key = client_url.split(":")[-1].rstrip("/v1").rstrip("/")
            metrics[f"client/{short_key}/rollouts"] = rollouts
            metrics[f"client/{short_key}/tokens"] = self.stats.client_tokens[client_url]
            client_time = self.stats.client_time[client_url]
            if client_time > 0:
                metrics[f"client/{short_key}/tokens_per_sec"] = (
                    self.stats.client_tokens[client_url] / client_time
                )

        return metrics
