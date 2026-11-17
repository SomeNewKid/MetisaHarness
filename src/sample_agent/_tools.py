"""Tools for the sample agent."""

from __future__ import annotations

from datetime import datetime

from metisa_common.mcp_helper import with_output_schema

_TOOL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "year": {"type": "integer"},
        "month": {"type": "integer"},
        "day": {"type": "integer"},
        "day_of_week": {"type": "integer"},
        "hour": {"type": "integer"},
        "minutes": {"type": "integer"},
        "seconds": {"type": "integer"},
        "utc_offset_minutes": {"type": "integer"},
    },
    "required": [
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "minutes",
        "seconds",
        "utc_offset_minutes",
    ],
    "additionalProperties": False,
}


@with_output_schema(_TOOL_RESULT_SCHEMA)
def get_current_date_time() -> dict[str, object]:
    """
    Return the current local date, time, day of the week, and UTC offset.

    Day of the week is represented as an integer, where Monday is 1 and Sunday is 7.
    """
    print("Call to get_current_date_time")
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_minutes = int(offset.total_seconds() / 60) if offset else 0
    return {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "day_of_week": now.isoweekday(),
        "hour": now.hour,
        "minutes": now.minute,
        "seconds": now.second,
        "utc_offset_minutes": offset_minutes,
    }
