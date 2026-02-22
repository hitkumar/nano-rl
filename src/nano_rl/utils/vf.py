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

# Required to preserve trajectory data in RolloutOutput
REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]


async def generate_group(
    client: vf.ClientConfig,
    env: vf.Environment,
    model_name: str,
    example: dict[str, Any],
    rollouts_per_example: int,
    sampling_args: dict[str, Any],
) -> list[vf.RolloutOutput]:
    """Generate a group of rollouts for an example, each RolloutOutput corresponds to one rollout"""
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]

    return await env.run_group(
        group_inputs,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        state_columns=REQUIRED_STATE_COLUMNS,
    )


async def generate_rollout(
    client: vf.ClientConfig,
    env: vf.Environment,
    model_name: str,
    example: dict[str, Any],
    sampling_args: dict[str, Any],
) -> vf.RolloutOutput:
    """Asynchronously generate a single rollout for an example and score it"""
    rollout_input = vf.RolloutInput(**example)
    return await env.run_rollout(
        rollout_input,
        client,
        model_name,
        sampling_args,
        state_columns=REQUIRED_STATE_COLUMNS,
    )


def get_completion_len(output: vf.RolloutOutput) -> int:
    """Total completion tokens across all turns."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"])


def get_prompt_len(output: vf.RolloutOutput) -> int:
    """Length of the initial prompt (first turn only)."""
    return len(output["trajectory"][0]["tokens"]["prompt_ids"])


def get_seq_len(output: vf.RolloutOutput) -> int:
    """Total sequence length. The last step's prompt contains the full conversation
    history, so prompt + completion of the last step gives the total token count."""
    last = output["trajectory"][-1]["tokens"]
    return len(last["prompt_ids"]) + len(last["completion_ids"])
