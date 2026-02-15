"""
Verifiers helpers
Verifiers helps us define RL environments and use existing ones.
RL environment is composed of 3 parts
- Dataset (list of questions and ground truth)
- Parser (extracts answer from model output)
- Rubric (scores the answer)
- rollout() produces model answer

# Environment defines:
dataset = [{"question": "What is 2+2?", "answer": "4"}, ...]
parser = XMLParser()  # extracts <answer>X</answer> from response
rubric = Rubric()     # reward = 1.0 if extracted == ground_truth else 0.0

# rollout() does:
prompt = format(question)           # "What is 2+2?"
response = await client.chat(...)   # Model says "The answer is <answer>4</answer>"
extracted = parser.parse(response)  # "4"
reward = rubric.score(extracted, ground_truth)  # 1.0

"""

from typing import Any

import verifiers as vf
from nano_rl.orchestrator.utils import get_semaphore
from openai import AsyncOpenAI


async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict[str, Any],
    rollouts_per_example: int,
    sampling_args: dict[str, Any],
) -> list[vf.State]:
    """Generate a group of rollouts for an example, each vf.State corresponds to one rollout"""
    semaphore = await get_semaphore()
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]

    states = await env.run_group(
        group_inputs=group_inputs,
        client=client,
        model=model_name,
        gen_sampling_args=sampling_args,
        gen_sem=semaphore,
        score_sem=semaphore,
    )
    return states


async def generate_rollout(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict[str, Any],
    sampling_args: dict[str, Any],
) -> vf.State:
    """Asynchronously generate a single rollout for an example and score it"""
    semaphore = await get_semaphore()

    rollout_input = vf.RolloutInput(**example)
    state = await env.run_rollout(
        semaphore, rollout_input, client, model_name, sampling_args
    )
    await env.rubric.score_rollout(state, score_sem=semaphore)
    return state


def get_completion_len(state: vf.State) -> int:
    """Total completion tokens across all turns."""
    return sum(len(step["tokens"]["completion_ids"]) for step in state["trajectory"])


def get_prompt_len(state: vf.State) -> int:
    """Length of the initial prompt (first turn only)."""
    return len(state["trajectory"][0]["tokens"]["prompt_ids"])


def get_seq_len(state: vf.State) -> int:
    """Total sequence length. The last step's prompt contains the full conversation
    history, so prompt + completion of the last step gives the total token count."""
    last = state["trajectory"][-1]["tokens"]
    return len(last["prompt_ids"]) + len(last["completion_ids"])
