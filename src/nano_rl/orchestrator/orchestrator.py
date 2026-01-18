"""Main orchestrator loop"""

import asyncio
import random

import uvloop
import verifiers as vf
from nano_rl.orchestrator.advantage import compute_advantages
from nano_rl.orchestrator.config import AdvantageConfig, OrchestratorConfig
from nano_rl.orchestrator.scheduler import Scheduler
from nano_rl.orchestrator.utils import set_semaphore
from nano_rl.transport import setup_training_batch_sender, TrainingBatch, TrainingSample
from nano_rl.utils.client import check_health, setup_admin_client, setup_client
from nano_rl.utils.logger import get_logger, setup_logger
from nano_rl.utils.pydantic_config import parse_argv
from nano_rl.utils.vf import get_completion_len


def state_to_training_sample(state: vf.State, advantage: float) -> TrainingSample:
    tokens = state["trajectory"][0]["tokens"]
    return TrainingSample(
        prompt_ids=tokens["prompt_ids"],
        prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
        completion_ids=tokens["completion_ids"],
        completion_mask=[bool(i) for i in tokens["completion_mask"]],
        completion_logprobs=tokens["completion_logprobs"],
        advantage=advantage,
    )


def setup_envs(config: OrchestratorConfig) -> list[vf.Environment]:
    # setup verifiers env from the config
    envs = []
    for env_config in config.env:
        env = vf.load_environment(env_config.id, **env_config.args)
        envs.append(env)
    return envs


async def orchestrate(config: OrchestratorConfig) -> None:
    """Main orchestration loop"""
    logger = get_logger()
    logger.info("Starting orchestrator")
    # limits the number of inference/scoring calls
    await set_semaphore(config.max_concurrent)

    # used for inference server completions calls
    client = setup_client(config.client)
    # used for admin endpoints (eg. health check, update_weights)
    admin_client = setup_admin_client(config.client)

    sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)
    logger.info("Waiting for inference server")
    await check_health(admin_client)

    envs = setup_envs(config)
    if config.seed is not None:
        random.seed(config.seed)

    # create scheduler
    scheduler = Scheduler(config, client, admin_client, envs, sender)

    # start weight polling in background while the main loop runs.
    # Both main task and this one take turns at the await points on the same thread
    update_task = asyncio.create_task(scheduler.update_policy_loop())

    step = 0
    max_steps = config.max_steps or float("inf")
    try:
        while step < max_steps:
            logger.info(f"Running Step {step}")
            states = await scheduler.generate_batch(step)
            if not states:
                logger.warning(f"Step {step} no valid states generated, retrying")
                continue

            rewards = [s["reward"] for s in states]
            completion_lens = [get_completion_len(s) for s in states]
            # compute advantages in groups of rollouts_per_example
            advantages = compute_advantages(
                rewards=rewards,
                completion_lens=completion_lens,
                rollouts_per_example=config.rollouts_per_example,
                advantage_config=config.advantage,
            )
            samples = [
                state_to_training_sample(state, advantage)
                for (state, advantage) in zip(states, advantages)
            ]

            # filter samples that are too long, maybe this filtering should be in scheduler
            samples = [
                s
                for s in samples
                if len(s.completion_ids) + len(s.prompt_ids) <= config.seq_len
            ]
            if not samples:
                logger.warning(
                    f"Step {step} all samples filtered by seq_len, generate again"
                )
                continue

            batch = TrainingBatch(
                examples=samples,
                temperature=config.sampling.temperature,
                step=step,
                ckpt_step=scheduler.current_weight_step,
            )

            # writes batch to a file from where trainer can read it
            batch_save_path = sender.send(batch)

            avg_reward = sum(rewards) / len(rewards)
            logger.info(
                f"Step {step} wrote {len(samples)} samples at {batch_save_path}, avg_reward={avg_reward:.3f}"
            )
            step += 1

    finally:
        # stop update_policy_loop
        scheduler.stop()
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass


def main():
    # Initialize the logger
    config = parse_argv(OrchestratorConfig)
    setup_logger(
        config.log.level,
        log_file=(
            config.output_dir / "logs" / "orchestrator.log" if config.log.file else None
        ),
    )
    uvloop.install()

    asyncio.run(orchestrate(config))


if __name__ == "__main__":
    main()
