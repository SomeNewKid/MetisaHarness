"""Tools for the sample agent."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from metisa_common.mcp_helper import with_output_schema

_LOGGER = logging.getLogger(__name__)

_DATE_TIME_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "year": {"type": "integer"},
        "month": {"type": "integer"},
        "day": {"type": "integer"},
        "hour": {"type": "integer"},
        "minutes": {"type": "integer"},
        "seconds": {"type": "integer"},
        "day_of_week": {"type": "integer"},
        "day_of_week_name": {"type": "string"},
        "utc_offset_minutes": {"type": "integer"},
    },
    "required": [
        "year",
        "month",
        "day",
        "hour",
        "minutes",
        "seconds",
        "day_of_week",
        "day_of_week_name",
        "utc_offset_minutes",
    ],
    "additionalProperties": False,
}

_PRODUCT_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "product_code": {"type": "string"},
        "product_name": {"type": "string"},
    },
    "required": ["product_code", "product_name"],
    "additionalProperties": False,
}

_PRODUCT_NAMES = {
    "LAMP": "  desk lamp  ",
    "CHAIR": "  office chair  ",
    "INTERNAL": "  internal prototype   ",
}


@with_output_schema(_DATE_TIME_RESULT_SCHEMA)
def get_current_date_time() -> dict[str, object]:
    """
    Return the current local date, time, day of the week, and UTC offset.

    Day of the week is represented as an integer, where Monday is 1 and Sunday is 7.
    """
    _LOGGER.debug("get_current_date_time request")
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_minutes = int(offset.total_seconds() / 60) if offset else 0

    result: dict[str, object] = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minutes": now.minute,
        "seconds": now.second,
        "day_of_week": now.isoweekday(),
        "day_of_week_name": now.strftime("%A"),
        "utc_offset_minutes": offset_minutes,
    }

    _LOGGER.debug(
        "get_current_date_time response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result


@with_output_schema(_PRODUCT_RESULT_SCHEMA)
def get_product_details(
    product_code: str,
) -> dict[str, object]:
    """Look up a product using its product code."""
    _LOGGER.debug("get_product_details request")

    result: dict[str, object] = {
        "product_code": product_code,
        "product_name": _PRODUCT_NAMES.get(product_code, "  unknown product   "),
    }

    _LOGGER.debug(
        "get_product_details response: %s",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    return result
