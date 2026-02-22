"""
Inspect tool-calling environments in verifiers.
Runs a multi-turn rollout and prints tool calls + tool responses per turn.

Usage:
    1. Start the inference server with tool calling enabled:
       uv run inference --model.name Qwen/Qwen3-0.6B \
           --model.enable-auto-tool-choice true --model.tool-call-parser hermes

    2. Run this script:
       uv run python src/nano_rl/utils/vf_tool_env_inspect.py
"""

import asyncio

import verifiers as vf
from openai import AsyncOpenAI


def multiturn_tool_call_reward(completion, info):
    """Check tool calls across all messages, not just the last one."""
    called = sorted(
        tc["function"]["name"]
        for msg in completion
        for tc in (msg.get("tool_calls") or [])
    )
    return 1.0 if called == sorted(info["tool_names"]) else 0.0


def print_message(msg):
    role = msg["role"]
    content = msg.get("content", "")
    if role == "tool":
        print(f"    [{role}] (id={msg.get('tool_call_id', '')}) {content}")
    elif msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            print(f"    [{role}] -> {fn['name']}({fn['arguments']})  [id={tc['id']}]")
    else:
        print(f"    [{role}] {content}")


async def main():
    env = vf.load_environment("primeintellect/tool-test")
    env.max_turns = 2
    env.rubric.funcs = [multiturn_tool_call_reward]
    env.rubric.weights = [1.0]

    dataset = env.get_dataset()
    sample = dataset[0]
    print(f"Tools: {[t.__name__ for t in env.tools]}")
    print(f"Prompt: {sample['prompt'][-1]['content']}")
    print(f"Expected: {sample['info']['tool_names']}")

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    state = await env.run_rollout(
        asyncio.Semaphore(1),
        vf.RolloutInput(**sample),
        client,
        "Qwen/Qwen3-0.6B",
        {"temperature": 1.0, "max_tokens": 768, "logprobs": True, "extra_body": {"return_token_ids": True}},
    )
    await env.rubric.score_rollout(state, score_sem=asyncio.Semaphore(1))

    print(f"\nReward: {state['reward']}")
    print(f"Turns: {len(state['trajectory'])}")

    for i, step in enumerate(state["trajectory"]):
        tokens = step["tokens"]
        print(f"\n{'─' * 60}")
        print(f"Turn {i}  (prompt: {len(tokens['prompt_ids'])} tokens, completion: {len(tokens['completion_ids'])} tokens)")
        print(f"  Prompt:")
        for msg in step["prompt"]:
            print_message(msg)
        print(f"  Completion:")
        for msg in step["completion"]:
            print_message(msg)


if __name__ == "__main__":
    asyncio.run(main())
