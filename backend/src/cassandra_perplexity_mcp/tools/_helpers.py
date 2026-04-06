"""Shared helpers for tool modules."""

from __future__ import annotations

from fastmcp.server.context import Context

from cassandra_perplexity_mcp.clients.perplexity import PerplexityClient

_fallback_client: PerplexityClient | None = None


def set_fallback_client(client: PerplexityClient) -> None:
    global _fallback_client  # noqa: PLW0603
    _fallback_client = client


def resolve_perplexity_client(ctx: Context) -> PerplexityClient:
    """Get the Perplexity client from lifespan context or fallback."""
    try:
        return ctx.request_context.lifespan_context["perplexity_client"]
    except (AttributeError, KeyError, TypeError):
        pass
    if _fallback_client is not None:
        return _fallback_client
    raise RuntimeError("No PerplexityClient available — check lifespan or fallback")
