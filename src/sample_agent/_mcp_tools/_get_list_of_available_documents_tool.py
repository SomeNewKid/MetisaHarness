"""Provide the tool which uses semantic search to get chunks from a document."""

from __future__ import annotations

from metisa_common.mcp_helper import with_output_schema

_TOOL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["chunks"],
    "additionalProperties": False,
}


@with_output_schema(_TOOL_RESULT_SCHEMA)
def get_list_of_available_documents() -> dict[str, object]:
    """List available company documents and their filenames.

    Use this to discover filenames before retrieving document content.
    """
    print("Call to get_list_of_available_documents()")

    result: dict[str, object] = {
        "files": [
            "Employee handbook.pdf",
        ]
    }

    return result
