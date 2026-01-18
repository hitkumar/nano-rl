"""
Coordinates rollout generation with weight updates in an async RL training loop.
"""

import asyncio
import random
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
from nano_rl.utils.vf import generate_group  # Rollout generation
from openai import AsyncOpenAI


class Scheduler:
    def __init__(
        self,
        config: OrchestratorConfig,
        client: AsyncOpenAI,
        admin_client: AsyncClient,
        envs: list[vf.Environment],
        # unused for now, we directly get the examples from the envs
        # examples_iter,  # iterator yielding training examples (prompts)
        sender: TrainingBatchSender,  # sends completed rollout batches to trainer.
    ):
        self.config = config
        self.client = client
        # used to update weights, do health checks.
        self.admin_client = admin_client
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
            await update_weights(self.admin_client, weights_path)
            self.current_weight_step = latest_step
            self.logger.info(f"Updated to weights step: {latest_step}")
            # possibly unblock batch generation
            self.checkpoint_ready.set()

    async def update_policy_loop(self) -> None:
        """Poll for new weights and update inference server"""
        while not self._stop:
            await self._update_policy()
            await asyncio.sleep(0.5)  # Poll every 500ms

    async def _wait_for_checkpoint(self, required_step: int) -> None:
        """Wait until a checkpoint at or after required_step is available"""
        while self.current_weight_step < required_step:
            self.logger.info(
                f"Waiting for ckpt to become available, currently at {self.current_weight_step}, need {required_step}"
            )
            self.checkpoint_ready.clear()  # set this to false
            await self.checkpoint_ready.wait()

    async def schedule_group_rollout(
        self, env: vf.Environment, example: dict[str, Any]
    ) -> list[vf.State]:
        states = await generate_group(
            client=self.client,
            env=env,
            model_name=self.config.model.name,
            example=example,
            rollouts_per_example=self.config.rollouts_per_example,
            sampling_args=self.sampling_args,
        )
        return states

    async def generate_batch(self, step: int) -> list[vf.State]:
        """Generates a batch of rollouts for training step"""
        self.step = step
        if self.config.max_async_level > 0:  # 0 means disabled
            min_required_step = self.step - self.config.max_async_level
            if self.current_weight_step < min_required_step:
                await self._wait_for_checkpoint(min_required_step)

        examples_needed = self.config.batch_size // self.config.rollouts_per_example
        tasks = []

        for i in range(examples_needed):
            env = self.envs[i % len(self.envs)]  # round robin across environments
            dataset = env.get_dataset()
            example = random.choice(dataset)
            task = asyncio.create_task(self.schedule_group_rollout(env, example))
            tasks.append(task)

        # wait for all tasks to complete (they run in parallel)
        results = await asyncio.gather(*tasks)
        all_states = []
        for result in results:
            all_states.extend(result)

        return all_states

    def stop(self):
        self._stop = True

    def get_metrics(self) -> dict[str, float]:
        """Return scheduler metrics for logging."""
        return {
            "scheduler/async_level": self.async_level,
            "scheduler/weight_step": self.current_weight_step,
            "scheduler/train_step": self.step,
        }
