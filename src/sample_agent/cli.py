"""Command-line interface for the application."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import win32api
import win32con
import win32security
from mcp import Client
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind

from metisa_common.specification_helper import (
    load_specification,
)
from metisa_common.specification_models import (
    Capability,
    MetisaSpecification,
)
from metisa_harness import (
    AgentDefinition,
    AgentHarness,
    LifecycleEvent,
)

from ._guardrails import (
    create_product_guardrail_plugin,
)
from ._helper import (
    print_elapsed_time,
    print_failure,
    print_success,
)
from ._mcp_server import (
    get_mcp_server,
)
from ._models import (
    ProductDetails,
)
from ._telemetry import (
    initialize_tracing,
)
from ._tools import (
    get_current_date_time,
    get_email_address,
    get_phone_number,
    get_product_details,
    get_robot_battery_charge,
    get_robot_motor_temperature,
    issue_refund,
)

_VERBOSE_LOGGING = False
_PRINT_LIFECYCLE_EVENTS = False

_root_directory = Path(__file__).parent
_metisa_json_path = _root_directory / "metisa.toml"
if not _metisa_json_path.is_file():
    raise FileNotFoundError("Sample Agent missing a metisa.toml file.")
specification: MetisaSpecification = load_specification(_metisa_json_path)


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the sample agent workload."""
    run_identifier, host_run_directory = _create_run_identifier_and_directory()
    host_output_path = _create_output_directory(host_run_directory, "output")

    user_sid = _get_launching_user_sid()
    if not user_sid:
        raise RuntimeError("METISA_USER_SID environment variable is not set.")

    prompt = _get_first_argument(argv)

    provider = None
    exporter = None

    os.environ["METISA_RUN_ID"] = run_identifier
    os.environ["OTEL_SERVICE_NAME"] = "sample_agent"

    default_logging_level = logging.DEBUG if _VERBOSE_LOGGING else logging.WARNING

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("metisa_harness").setLevel(default_logging_level)
    logging.getLogger("sample_agent").setLevel(default_logging_level)

    if Capability.OTEL_COLLECTOR in specification.capabilities:
        provider, exporter = initialize_tracing()

    # OpenTelemetry supports instrumentation without an initialized SDK provider.
    # Calls such as trace.get_tracer() and start_as_current_span()
    # do not record or export spans until the provider is configured.
    try:
        tracer = trace.get_tracer("sample_agent")

        attributes = {
            "run_id": run_identifier,
            "user_sid": user_sid,
        }

        with tracer.start_as_current_span(
            "sample_agent.run",
            attributes=attributes,
            context=Context(),
        ):
            import asyncio

            asyncio.run(
                _run_agent(
                    specification,
                    host_output_path,
                    prompt,
                    run_identifier,
                )
            )

        return 0

    except Exception as error:
        print_failure(str(error))
        return 1

    finally:
        if provider is not None:
            try:
                if exporter is not None:
                    spans = exporter.get_finished_spans()
                    documents = [json.loads(span.to_json()) for span in spans]
                    content = json.dumps(documents, indent=2)

                    trace_path = (
                        host_run_directory / ".logs" / "telemetry" / "traces.json"
                    )
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(content, encoding="utf-8")
            finally:
                provider.shutdown()


async def _run_agent(
    specification: MetisaSpecification,
    host_output_path: Path,
    prompt: str,
    run_identifier: str,
) -> None:
    tracer = trace.get_tracer("sample_agent.tests")

    test_name = "tracing"

    with tracer.start_as_current_span(
        f"tests.call {test_name}",
        kind=SpanKind.CLIENT,
        attributes={
            "metisa.test.name": test_name,
            "metisa.reason": "testing tracing",
        },
    ):
        # Example prompts:
        # - Look up product code "chair". (Tests guardrails)
        # - How many company values do we have? (Tests RAG)
        # - What is the current time? (Tests tools)
        # - What is the email address and phone number for Claire? (Agents as tools)
        # - Issue a refund for $75.40. (Human in the loop approval)
        #   (over $50 requires approval, and over $100 is automatically rejected)
        # - Can the robot named Bolt enter the race? (Test skills.)
        # - Can the robot named Alice be placed in storage? (Test skills with scripts)
        default_prompt = "Can the robot named Alice be placed in storage?"

        prompt_count = 0
        if not prompt:
            if Capability.INTERACTIVE not in specification.capabilities:
                prompt = default_prompt
        while not prompt:
            prompt_count += 1
            if prompt_count > 3:
                raise RuntimeError("Too many empty prompts entered.")
            prompt = input("Enter prompt: ")

        models = [
            "gpt-4o",
            "granite4.1:3b",
            # "granite4.1:8b",
            # "gemma4:12b",
        ]

        for model in models:
            await _run_agent_model(model, prompt, run_identifier)


async def _run_agent_model(
    model_name: str,
    prompt: str,
    run_identifier: str,
) -> None:
    mcp_server = get_mcp_server()

    guardrail_plugin = create_product_guardrail_plugin()

    email_agent = AgentDefinition(
        name="email_agent",
        description="Retrieve email addresses for users.",
        model=model_name,
        instructions="You provide the email address for the specified user.",
        tools=[get_email_address],
    )

    phone_agent = AgentDefinition(
        name="phone_agent",
        description="Retrieve phone numbers for users.",
        model=model_name,
        instructions="You provide the phone number for the specified user.",
        tools=[get_phone_number],
    )

    async with Client(mcp_server) as mcp_client:
        main_agent = AgentDefinition(
            name="sample_agent",
            description="Example agent",  # Description really only used for sub-agents
            model=model_name,
            instructions=dedent("""
                Answer the user's question as simply as possible.
                Provide the answer as plain text, and not as Markdown or JSON.
                Discover the date and time with get_current_date_time.
                Discover available filenames with get_list_of_available_documents,
                then retrieve relevant excerpts with get_semantic_chunks_from_document.
                Check the documents before asking the user to supply a filename.
                For product questions, call get_product_details,
                passing the product code exactly as supplied by the user,
                including its capitalization and surrounding spaces.
                For robot inspections, identify the matching skill and 
                follow its checklist.  If you cannot load the checklist or
                a required reference, stop and explain the failure.
                Do not guess the answer if it is not available in the documents 
                or through the tools.
                If you cannot find the answer, explain what documents were checked and
                which tools were used, before stating that the answer is not available.
                """),
            tools=[
                get_current_date_time,
                get_product_details,
                get_robot_battery_charge,
                get_robot_motor_temperature,
                issue_refund,
            ],
            mcp_clients=[mcp_client],
            sub_agents=[email_agent, phone_agent],
            guardrail_plugins=(guardrail_plugin,),
            skills_directory=_root_directory / "skills",
            # output_type=ProductDetails,
        )

        print()
        print("-" * 20)
        print(model_name)
        print(prompt, flush=True)
        with print_elapsed_time():
            try:
                result = await AgentHarness.run(
                    main_agent,
                    run_identifier=run_identifier,
                    input=prompt,
                    event_handler=_print_lifecycle_event,
                )
                if result.final_output:
                    if main_agent.output_type is ProductDetails:
                        if isinstance(result.final_output, ProductDetails):
                            product_code = result.final_output.product_code
                            product_name = result.final_output.product_name
                            print_success(
                                f"ProductDetails '{product_code}', '{product_name}'"
                            )
                        else:
                            print_failure("Final output of wrong type.")
                    else:
                        print_success(result.final_output)
                else:
                    raise RuntimeError("No final text returned.")
            except Exception as error:
                print_failure(str(error))


def _create_run_identifier_and_directory() -> tuple[str, Path]:
    suffix = 0
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_identifier = f"run-{timestamp}"
    path: Path | None = None
    while not path and suffix <= 99:
        try:
            path = _create_run_directory(run_identifier)
        except FileExistsError:
            suffix += 1
            run_identifier = f"run-{timestamp}-{suffix:02d}"
    if not path:
        raise RuntimeError("Cannot create unique run identifier.")
    return run_identifier, path


def _create_run_directory(
    run_identifier: str,
) -> Path:
    run_directory = Path.cwd() / ".runs" / run_identifier
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _create_output_directory(
    host_run_directory: Path,
    output_dir_name: str,
) -> Path:
    output_directory = host_run_directory / output_dir_name
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def _get_launching_user_sid() -> str:
    process_handle = win32api.GetCurrentProcess()
    token_handle = win32security.OpenProcessToken(
        process_handle,
        win32con.TOKEN_QUERY,
    )

    try:
        user_sid, _attributes = win32security.GetTokenInformation(
            token_handle,
            win32security.TokenUser,
        )
        sid_string = win32security.ConvertSidToStringSid(user_sid)
    finally:
        win32api.CloseHandle(token_handle)

    return sid_string


def _get_first_argument(
    argv: list[str] | None = None,
) -> str:
    """Get the first command-line argument, or an empty string if none are provided."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return ""

    return args[0]


async def _print_lifecycle_event(
    event: LifecycleEvent,
) -> None:
    if not _PRINT_LIFECYCLE_EVENTS:
        return

    details = [event.agent_name, event.kind]

    if event.turn is not None:
        details.append(f"turn={event.turn}")

    if event.turn is not None:
        details.append(f"turn={event.turn}")

    if event.tool_name is not None:
        details.append(f"tool={event.tool_name}")

    if event.error_type is not None:
        details.append(f"error={event.error_type}")

    description = " ".join(details)
    print(f"{description} [{event.elapsed_seconds:.3f}s]", flush=True)
