"""
Util file to play around with the envs in verifiers
"""

import asyncio

import verifiers as vf
from openai import AsyncOpenAI


async def main():
    env = vf.load_environment("primeintellect/alphabet-sort", min_turns=2, max_turns=3)
    dataset = env.get_dataset()
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"First sample: {dataset[0]}")
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    state = await env.run_rollout(
        asyncio.Semaphore(1),
        vf.RolloutInput(**dataset[0]),
        client,
        "Qwen/Qwen3-0.6B",
        {
            "temperature": 1.0,
            "max_tokens": 768,
            "logprobs": True,
            "extra_body": {"return_token_ids": True},
        },
    )
    await env.rubric.score_rollout(state, score_sem=asyncio.Semaphore(1))
    print(f"Reward: {state['reward']}")
    print(f"Turns: {len(state['trajectory'])}")
    for i, step in enumerate(state["trajectory"]):
        tokens = step["tokens"]
        print(f"\n--- Turn {i} ---")
        print(f"  Prompt tokens: {len(tokens['prompt_ids'])}")
        print(f"  Completion tokens: {len(tokens['completion_ids'])}")
        print(f"  Prompt: {step['prompt'][-1] if step['prompt'] else 'N/A'}")
        print(
            f"  Completion: {step['completion'][-1] if step['completion'] else 'N/A'}"
        )


if __name__ == "__main__":
    asyncio.run(main())
