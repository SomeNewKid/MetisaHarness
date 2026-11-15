"""Configure tracing for the sample workload."""

from __future__ import annotations

import os

import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON


def initialize_tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Initialize the workload's tracing provider."""
    service_name = os.environ["OTEL_SERVICE_NAME"]
    run_identifier = os.environ["METISA_RUN_ID"]

    # Identifies the service and application run.
    resource = Resource.create(
        {
            "service.name": service_name,
            "metisa.run_id": run_identifier,
        }
    )

    # Sends spans to the Collector using HTTP.
    session = requests.Session()
    session.trust_env = False

    exporter = InMemorySpanExporter()

    # Exports completed spans in batches.
    processor = SimpleSpanProcessor(exporter)

    # Owns the tracing configuration.
    provider = TracerProvider(
        resource=resource,
        sampler=ALWAYS_ON,  # Records all spans for this initial implementation
    )
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    return provider, exporter
