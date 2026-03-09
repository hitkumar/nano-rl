"""
Math + Python tool-calling environment for multi-turn RL training.

The model solves math problems by calling a python() tool to run code,
then gives a final \\boxed{} answer. Each python() call is a fresh
subprocess (no persistent state across calls).

Usage:
    from nano_rl.envs.math_python import load_environment
    env = load_environment()
"""

import ast
import subprocess
import textwrap

import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def _jupyterize(src: str) -> str:
    """Transform code so trailing expressions are printed, like Jupyter."""
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError:
        return src

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        body_code = ast.unparse(ast.Module(tree.body, []))
        expr_code = ast.unparse(last.value)
        return f"{body_code}\n_ = {expr_code}\nif _ is not None: print(_)"
    return src


async def python(code: str) -> str:
    """Run Python code and return the output. Standard library + numpy, sympy, scipy available.

    Args:
        code: A block of Python code.

    Returns:
        The stdout output or error message.
    """
    try:
        result = subprocess.run(
            ["python", "-c", _jupyterize(code)],
            timeout=10,
            text=True,
            capture_output=True,
        )
        output = result.stdout.strip()[:2000]
        error = result.stderr.strip()[:2000]
        if error:
            return error
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"


SYSTEM_PROMPT = (
    "Use Python to solve the problem. "
    "Call the python tool to run code and see the output. "
    "In addition to the Python standard library, you have access to: numpy, sympy, scipy. "
    "Give your final answer inside \\boxed{}."
)


def load_environment(
    dataset_name: str = "math",
    dataset_split: str | None = None,
    num_examples: int = -1,
    max_turns: int = 5,
) -> vf.ToolEnv:
    """Load the math-python environment.

    Args:
        dataset_name: Dataset to use (math, gsm8k, math500, etc.)
        dataset_split: Dataset split. None uses the default for the dataset.
        num_examples: Number of examples to use. -1 for all.
        max_turns: Maximum number of tool-calling turns.
    """
    kwargs = {}
    if dataset_split is not None:
        kwargs["split"] = dataset_split
    dataset = load_example_dataset(dataset_name, n=num_examples if num_examples > 0 else None, **kwargs)

    parser = vf.Parser(extract_fn=extract_boxed_answer)
    rubric = vf.MathRubric(parser=parser)

    return vf.ToolEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=max_turns,
    )
