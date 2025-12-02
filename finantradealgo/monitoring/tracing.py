"""OpenTelemetry tracing setup and helpers."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Mapping

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    OTLPSpanExporter,
    SpanExporter,
)
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .metrics_collector import MonitoringConfig


def _build_resource(config: MonitoringConfig) -> Resource:
    service_name = config.service_name or "finantrade_service"
    attributes: dict[str, Any] = {
        ResourceAttributes.SERVICE_NAME: service_name,
    }
    if config.environment:
        attributes["service.environment"] = config.environment
    if config.labels:
        for key, value in config.labels.items():
            attributes[f"service.label.{key}"] = str(value)
    return Resource.create(attributes)


def _build_exporter(config: MonitoringConfig) -> SpanExporter:
    if config.otlp_endpoint:
        return OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=config.otlp_insecure,
        )
    return ConsoleSpanExporter()


def initialize_tracing(config: MonitoringConfig) -> TracerProvider:
    """
    Configure OpenTelemetry tracing for the service.

    Returns the initialized TracerProvider so the caller can reuse it.
    """
    resource = _build_resource(config)
    provider = TracerProvider(resource=resource)
    exporter = _build_exporter(config)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


def instrument_fastapi_app(app: FastAPI, tracer_provider: TracerProvider | None = None) -> None:
    """
    Attach FastAPI instrumentation for request spans.
    Call after tracer_provider is initialized.
    """
    if tracer_provider:
        trace.set_tracer_provider(tracer_provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())


def trace_span(span_name: str, **base_attributes: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to wrap sync/async callables with an OpenTelemetry span.

    Example:
        @trace_span("backtest.run", strategy="my-strategy")
        def run_backtest(...):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tracer = trace.get_tracer(__name__)

        if asyncio.iscoroutinefunction(func):  # type: ignore[name-defined]

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(span_name) as span:
                    _apply_attributes(span, base_attributes, kwargs)
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                _apply_attributes(span, base_attributes, kwargs)
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def _apply_attributes(span: trace.Span, base_attrs: Mapping[str, Any], call_kwargs: Mapping[str, Any]) -> None:
    attrs: dict[str, Any] = dict(base_attrs)
    for key, value in call_kwargs.items():
        if key not in attrs and _is_primitive(value):
            attrs[key] = value
    for key, value in attrs.items():
        span.set_attribute(key, value)


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))
