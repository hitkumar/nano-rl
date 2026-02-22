"""
Inspect and evaluate tool-calling environments in verifiers.

Workaround: verifiers 0.1.11+ uses Pydantic messages where AssistantMessage.tool_calls
defaults to None (not absent). The upstream tool-test reward function uses
.get("tool_calls", []) which returns None instead of [] because getattr finds
the attribute. We override the reward function here to handle this.

Usage:
    Start the inference server with tool calling enabled:
        uv run inference --model.name Qwen/Qwen3-0.6B \
            --model.enable-auto-tool-choice true --model.tool-call-parser hermes

    Inspect a single rollout:
        uv run python src/nano_rl/utils/vf_tool_env_inspect.py inspect

    Batch eval (like prime eval run):
        uv run python src/nano_rl/utils/vf_tool_env_inspect.py eval -n 20 -r 4
        uv run python src/nano_rl/utils/vf_tool_env_inspect.py eval -n 20 -r 4 -m outputs/runs/tool_test_rl/weights/step_10
"""

import argparse
import asyncio
import random

import verifiers as vf


def tool_call_reward(completion, info):
    """Check tool calls across all messages, handles None tool_calls."""
    called = sorted(
        tc.name
        for msg in completion
        for tc in (getattr(msg, "tool_calls", None) or [])
    )
    return 1.0 if called == sorted(info["tool_names"]) else 0.0


def print_message(msg):
    role = msg.role
    content = msg.content or ""
    if role == "tool":
        print(f"    [{role}] (id={getattr(msg, 'tool_call_id', '')}) {content}")
    elif getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"    [{role}] -> {tc.name}({tc.arguments})  [id={tc.id}]")
    else:
        print(f"    [{role}] {content}")


def load_env():
    env = vf.load_environment("primeintellect/tool-test")
    # Replace the original reward function in the rubric.
    # env.rubric is a RubricGroup [original_rubric, ToolMonitorRubric, MultiTurnMonitorRubric].
    # The original rubric (rubrics[0]) has tool_call_reward_func which crashes on
    # Pydantic messages because .get("tool_calls", []) returns None instead of [].
    env.rubric.rubrics[0].funcs = [tool_call_reward]
    env.rubric.rubrics[0].weights = [1.0]
    return env


async def inspect(args):
    env = load_env()
    env.max_turns = 2
    dataset = env.get_dataset()
    sample = dataset[0]
    print(f"Tools: {[t.__name__ for t in env.tools]}")
    print(f"Prompt: {sample['prompt'][-1]['content']}")
    print(f"Expected: {sample['info']['tool_names']}")

    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.base_url,
        api_key_var="OPENAI_API_KEY",
    )
    output = await env.run_rollout(
        vf.RolloutInput(**sample),
        client,
        args.model,
        {"temperature": 1.0, "max_tokens": 768, "logprobs": True, "extra_body": {"return_token_ids": True}},
        state_columns=["trajectory"],
    )

    print(f"\nReward: {output['reward']}")
    print(f"Turns: {len(output['trajectory'])}")
    for i, step in enumerate(output["trajectory"]):
        tokens = step["tokens"]
        print(f"\n{'─' * 60}")
        print(f"Turn {i}  (prompt: {len(tokens['prompt_ids'])} tokens, completion: {len(tokens['completion_ids'])} tokens)")
        print(f"  Prompt:")
        for msg in step["prompt"]:
            print_message(msg)
        print(f"  Completion:")
        for msg in step["completion"]:
            print_message(msg)


async def eval_batch(args):
    env = load_env()
    eval_dataset = env.get_eval_dataset()
    examples = random.sample(list(eval_dataset), min(args.num_examples, len(eval_dataset)))

    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.base_url,
        api_key_var="OPENAI_API_KEY",
    )
    sampling_args = {"temperature": 1.0, "max_tokens": args.max_tokens}

    rewards_per_rollout = [[] for _ in range(args.rollouts_per_example)]
    for i, example in enumerate(examples):
        group_inputs = [vf.RolloutInput(**example) for _ in range(args.rollouts_per_example)]
        outputs = await env.run_group(
            group_inputs,
            client=client,
            model=args.model,
            sampling_args=sampling_args,
        )
        for r_idx, output in enumerate(outputs):
            rewards_per_rollout[r_idx].append(output["reward"])
        avg = sum(o["reward"] for o in outputs) / len(outputs)
        print(f"  [{i+1}/{len(examples)}] avg_reward={avg:.3f}")

    all_rewards = [r for rollout in rewards_per_rollout for r in rollout]
    avg_reward = sum(all_rewards) / len(all_rewards)
    std_reward = (sum((r - avg_reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

    print(f"\n--- Results ---")
    print(f"reward: avg={avg_reward:.3f}, std={std_reward:.3f}")
    for r_idx, rewards in enumerate(rewards_per_rollout):
        print(f"r{r_idx+1}: {rewards}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("-b", "--base-url", default="http://localhost:8000/v1")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect")

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("-n", "--num-examples", type=int, default=20)
    eval_parser.add_argument("-r", "--rollouts-per-example", type=int, default=4)
    eval_parser.add_argument("-t", "--max-tokens", type=int, default=768)

    args = parser.parse_args()
    if args.command == "inspect":
        asyncio.run(inspect(args))
    elif args.command == "eval":
        asyncio.run(eval_batch(args))


if __name__ == "__main__":
    main()
