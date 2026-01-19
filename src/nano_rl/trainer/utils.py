from datetime import timedelta
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from nano_rl.trainer.world import get_world
from nano_rl.utils.logger import get_logger
from nano_rl.utils.utils import format_num, format_time
from rich.console import Console
from rich.table import Table


DEFAULT_TIMEOUT = timedelta(seconds=300)


def setup_torch_distributed(
    timeout: timedelta = DEFAULT_TIMEOUT, enable_gloo: bool = False
):
    torch.cuda.set_device(get_world().local_rank)
    backend = None  # nccl by default
    if enable_gloo:
        get_logger().info("Using gloo backend for torch.distributed")
        backend = "cpu:gloo,cuda:nccl"
    dist.init_process_group(backend=backend, timeout=timeout)


def log0(*args, **kwargs):
    if get_world().rank == 0:
        get_logger().info(*args, **kwargs)


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """Print benchmark results in a table format"""
    world = get_world()
    if not world.is_master:
        return

    # Check required metrics exist
    required = ["perf/mfu", "perf/throughput", "time/step", "perf/peak_memory"]
    if not all(k in history and len(history[k]) > 1 for k in required):
        return

    # Check if broadcast time is available
    has_broadcast = "time/broadcast" in history and len(history["time/broadcast"]) > 1

    df_data = {
        "MFU": history["perf/mfu"],
        "Throughput": history["perf/throughput"],
        "Time/Step": history["time/step"],
        "Peak Memory": history["perf/peak_memory"],
    }
    if has_broadcast:
        df_data["Broadcast"] = history["time/broadcast"]

    df = pd.DataFrame(df_data)

    df = df.iloc[1:]  # Skip step 1 (warmup), show metrics from step 2 onwards
    if len(df) == 0:
        return

    console = Console()
    table = Table(title="Benchmark Results")
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center")

    # formatted dataframe
    formatted_df = pd.DataFrame()
    formatted_df["MFU"] = df["MFU"].apply(lambda x: f"{format_num(x, precision=2)}%")
    formatted_df["Throughput"] = df["Throughput"].apply(
        lambda x: f"{format_num(x, precision=2)} tok/s"
    )
    formatted_df["Time/Step"] = df["Time/Step"].apply(format_time)
    formatted_df["Peak Memory"] = df["Peak Memory"].apply(
        lambda x: f"{format_num(x, precision=1)} GiB"
    )
    if has_broadcast:
        formatted_df["Broadcast"] = df["Broadcast"].apply(format_time)

    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step + 1)] + [str(x) for x in row]))

    # Add separator row between steps and overall results
    table.add_row(*([""] * (len(formatted_df.columns) + 1)))

    # Calculate and add summary statistics row
    stats = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_stats = pd.DataFrame()
    formatted_stats["MFU"] = stats["MFU"].apply(
        lambda x: f"{format_num(x, precision=2)}%"
    )
    formatted_stats["Throughput"] = stats["Throughput"].apply(
        lambda x: format_num(x, precision=2)
    )
    formatted_stats["Time/Step"] = stats["Time/Step"].apply(format_time)
    if has_broadcast:
        formatted_stats["Broadcast"] = stats["Broadcast"].apply(format_time)

    # Get total GPU memory for percentage calculation
    if torch.cuda.is_available():
        total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    else:
        total_mem = 1

    # Build summary row with mean ± std [min, max] format
    # Handle columns in order: MFU, Throughput, Time/Step, Peak Memory, [Broadcast]
    summary_parts = []
    for col in ["MFU", "Throughput", "Time/Step"]:
        summary_parts.append(
            f"{formatted_stats[col]['mean']} ± {formatted_stats[col]['std']} "
            f"[{formatted_stats[col]['min']}, {formatted_stats[col]['max']}]"
        )
    # Peak Memory with special percentage format
    summary_parts.append(
        f"{format_num(stats['Peak Memory']['mean'], precision=1)} GiB "
        f"({stats['Peak Memory']['mean'] / total_mem * 100:.1f}%)"
    )
    if has_broadcast:
        summary_parts.append(
            f"{formatted_stats['Broadcast']['mean']} ± {formatted_stats['Broadcast']['std']} "
            f"[{formatted_stats['Broadcast']['min']}, {formatted_stats['Broadcast']['max']}]"
        )

    summary_row = ["Overall"] + summary_parts
    table.add_row(*summary_row)

    console.print(table)
