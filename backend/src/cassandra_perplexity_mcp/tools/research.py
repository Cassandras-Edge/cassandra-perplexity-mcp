"""Perplexity research tools — search, ask, and ask_reasoning with grounded responses."""

from __future__ import annotations

from typing import Literal, Optional

from fastmcp import FastMCP
from fastmcp.server.context import Context
from mcp.types import ToolAnnotations

from cassandra_perplexity_mcp.clients.perplexity import (
    RecencyOption,
    format_chat_response,
    format_search_results,
)
from cassandra_perplexity_mcp.config import Settings
from cassandra_perplexity_mcp.tools._helpers import resolve_perplexity_client


def register(mcp: FastMCP, settings: Settings) -> None:
    _ro = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True,
    )

    @mcp.tool(annotations=_ro)
    async def search(
        query: str,
        ctx: Context,
        max_results: int = 10,
        max_tokens_per_page: int = 1024,
        country: Optional[str] = None,
        recency: Optional[RecencyOption] = None,
        domain_filter: Optional[list[str]] = None,
        sources: Optional[Literal["web", "academic", "sec"]] = None,
    ) -> str:
        """Find and evaluate sources yourself. Returns URLs, titles, and snippets.

        Prefer this first to ground yourself before using ask.

        Args:
            query: Search query.
            max_results: Max results (1-20, default: 10).
            max_tokens_per_page: Max tokens per page (default: 1024).
            country: Two-letter country code to filter results (e.g., 'US', 'GB', 'DE').
            recency: Time filter — 'hour', 'today', 'day', 'yesterday', 'week',
                'last_week', 'month', 'last_month', 'year'.
            domain_filter: Filter by domain. Use '-' to exclude.
                Examples: ['github.com'], ['-reddit.com'].
            sources: Source type — 'web' (general), 'sec' (financial filings),
                'academic' (scholarly).
        """
        client = resolve_perplexity_client(ctx)
        data = await client.search(
            query=query,
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page,
            country=country,
            recency=recency,
            domain_filter=domain_filter,
            search_mode=sources,
        )
        return format_search_results(data)

    @mcp.tool(annotations=_ro)
    async def ask(
        query: str,
        ctx: Context,
        sources: Literal["web", "sec", "academic"] = "web",
        scope: Literal["standard", "extensive"] = "standard",
        thoroughness: Literal["quick", "detailed"] = "quick",
        recency: Optional[RecencyOption] = None,
        domain_filter: Optional[list[str]] = None,
        return_related_questions: bool = False,
        max_tokens: int = 5000,
    ) -> str:
        """Get AI-synthesized answers with web-grounded search.

        Args:
            query: Your question.
            sources: Source type — 'web' (general), 'sec' (financial filings),
                'academic' (scholarly).
            scope: Search breadth — 'standard' (normal) or 'extensive' (2x more sources).
            thoroughness: Content extraction — 'quick' (recommended) or 'detailed'
                (only if absolutely needed, prefer scope='extensive' instead).
            recency: Time filter — 'hour', 'today', 'day', 'yesterday', 'week',
                'last_week', 'month', 'last_month', 'year'.
            domain_filter: Filter by domain. Use '-' to exclude.
                Examples: ['github.com'], ['-reddit.com'].
            return_related_questions: Get follow-up questions.
            max_tokens: Max response length (default: 5000).
        """
        model = "sonar" if scope == "standard" else "sonar-pro"
        search_context_size = "low" if thoroughness == "quick" else "high"

        client = resolve_perplexity_client(ctx)
        data = await client.chat_completion(
            query=query,
            model=model,
            search_mode=sources,
            recency=recency,
            domain_filter=domain_filter,
            return_related_questions=return_related_questions,
            max_tokens=max_tokens,
            search_context_size=search_context_size,
        )
        return format_chat_response(data)

    @mcp.tool(annotations=_ro)
    async def ask_reasoning(
        query: str,
        ctx: Context,
        scope: Literal["standard", "extensive"] = "standard",
        thoroughness: Literal["quick", "detailed"] = "quick",
        recency: Optional[RecencyOption] = None,
        domain_filter: Optional[list[str]] = None,
        return_related_questions: bool = False,
        max_tokens: int = 5000,
    ) -> str:
        """Get answers with explicit step-by-step reasoning.

        Shows reasoning process with <think> sections. Use for multi-step problems
        and complex reasoning.

        Args:
            query: Your question.
            scope: Search breadth — 'standard' (normal) or 'extensive'
                (2x more sources, deeper reasoning).
            thoroughness: Content extraction — 'quick' (recommended) or 'detailed'
                (only if absolutely needed, prefer scope='extensive' instead).
            recency: Time filter — 'hour', 'today', 'day', 'yesterday', 'week',
                'last_week', 'month', 'last_month', 'year'.
            domain_filter: Filter by domain. Use '-' to exclude.
                Examples: ['github.com'], ['-reddit.com'].
            return_related_questions: Get follow-up questions.
            max_tokens: Max response length (default: 5000).
        """
        search_context_size = "low" if thoroughness == "quick" else "high"

        client = resolve_perplexity_client(ctx)
        data = await client.chat_completion(
            query=query,
            model="sonar-reasoning-pro",
            search_mode="web",
            recency=recency,
            domain_filter=domain_filter,
            return_related_questions=return_related_questions,
            max_tokens=max_tokens,
            search_context_size=search_context_size,
        )
        return format_chat_response(data)
