"""
General utility functions.

These are used across the codebase for consistent formatting of:
- Numbers (with K/M/B suffixes)
- Time durations (in appropriate units)
- Data transformations (row/column format conversion)
"""

from collections import defaultdict
from typing import Any


def format_time(time_in_seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Automatically selects appropriate unit:
    - < 1 minute: "X.XXs"
    - < 1 hour: "X.XXm"
    - >= 1 hour: "X.XXh"

    Examples:
        format_time(0.5)    → "0.50s"
        format_time(90)     → "1.50m"
        format_time(3700)   → "1.03h"
    """
    from datetime import timedelta

    td = timedelta(seconds=time_in_seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        total_hours = days * 24 + hours
        return f"{total_hours + minutes / 60:.2f}h"
    elif hours > 0:
        return f"{hours + minutes / 60:.2f}h"
    elif minutes > 0:
        return f"{minutes + seconds / 60:.2f}m"
    else:
        # Include microseconds for sub-second precision
        total_seconds = seconds + td.microseconds / 1_000_000
        return f"{total_seconds:.2f}s"


def format_num(num: float | int, precision: int = 2) -> str:
    """
    Format number with K/M/B suffix for readability.

    Examples:
        format_num(1234)        → "1.23K"
        format_num(1234567)     → "1.23M"
        format_num(1234567890)  → "1.23B"
        format_num(123)         → "123.00"
    """
    sign = "-" if num < 0 else ""
    num = abs(num)

    if num < 1e3:
        if isinstance(num, float):
            return f"{sign}{num:.{precision}f}"
        return f"{sign}{num}"
    elif num < 1e6:
        return f"{sign}{num / 1e3:.{precision}f}K"
    elif num < 1e9:
        return f"{sign}{num / 1e6:.{precision}f}M"
    else:
        return f"{sign}{num / 1e9:.{precision}f}B"


def to_col_format(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert row format (list of dicts) to column format (dict of lists).

    This is useful for converting monitor history to a format suitable
    for pandas DataFrames or the benchmark table.

    Example:
        Input:  [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        Output: {"a": [1, 3], "b": [2, 4]}
    """
    dict_of_lists: dict[str, list[Any]] = defaultdict(list)
    for row in list_of_dicts:
        for key, value in row.items():
            dict_of_lists[key].append(value)

    return dict_of_lists
