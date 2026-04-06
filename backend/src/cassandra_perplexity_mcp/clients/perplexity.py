"""Perplexity API client.

Wraps the Search and Chat Completions endpoints with recency/domain filtering.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = "https://api.perplexity.ai/search"
CHAT_ENDPOINT = "https://api.perplexity.ai/chat/completions"

RecencyOption = Literal[
    "hour", "today", "day", "yesterday", "week", "last_week", "month", "last_month", "year"
]

_RECENCY_PASSTHROUGH = {"hour", "day", "week", "month", "year"}


def _resolve_recency(recency: Optional[str]) -> dict:
    """Convert friendly recency value to Perplexity API parameters."""
    if not recency:
        return {}

    if recency in _RECENCY_PASSTHROUGH:
        return {"search_recency_filter": recency}

    today = date.today()
    fmt = "%m/%d/%Y"

    if recency == "today":
        return {"search_after_date_filter": today.strftime(fmt)}

    if recency == "yesterday":
        yesterday = today - timedelta(days=1)
        return {
            "search_after_date_filter": yesterday.strftime(fmt),
            "search_before_date_filter": today.strftime(fmt),
        }

    if recency == "last_week":
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(weeks=1)
        return {
            "search_after_date_filter": last_monday.strftime(fmt),
            "search_before_date_filter": this_monday.strftime(fmt),
        }

    if recency == "last_month":
        first_of_month = today.replace(day=1)
        first_of_last_month = (first_of_month - timedelta(days=1)).replace(day=1)
        return {
            "search_after_date_filter": first_of_last_month.strftime(fmt),
            "search_before_date_filter": first_of_month.strftime(fmt),
        }

    return {}


class PerplexityClient:
    """HTTP client for the Perplexity API."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._http = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self._http.aclose()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        max_results: int = 10,
        max_tokens_per_page: int = 1024,
        country: Optional[str] = None,
        recency: Optional[str] = None,
        domain_filter: Optional[list[str]] = None,
        search_mode: Optional[str] = None,
    ) -> dict:
        """Search endpoint — returns raw results with titles, URLs, snippets."""
        payload: dict = {
            "query": query,
            "max_results": max_results,
            "max_tokens_per_page": max_tokens_per_page,
        }

        if country:
            payload["country"] = country

        payload.update(_resolve_recency(recency))

        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        if search_mode:
            payload["search_mode"] = search_mode

        try:
            response = await self._http.post(
                SEARCH_ENDPOINT, json=payload, headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Search API error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    async def chat_completion(
        self,
        query: str,
        model: Literal["sonar", "sonar-pro", "sonar-reasoning-pro"] = "sonar",
        search_mode: Optional[str] = None,
        recency: Optional[str] = None,
        domain_filter: Optional[list[str]] = None,
        return_images: bool = False,
        return_related_questions: bool = False,
        max_tokens: int = 5000,
        search_context_size: Literal["low", "medium", "high"] = "medium",
        system_prompt: Optional[str] = "Be concise and factual. Cite sources. Avoid speculation.",
    ) -> dict:
        """Chat completions endpoint — AI-synthesized answers with citations."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "web_search_options": {
                "search_context_size": search_context_size,
            },
        }

        if search_mode:
            payload["search_mode"] = search_mode

        payload.update(_resolve_recency(recency))

        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        if return_images:
            payload["return_images"] = True

        if return_related_questions:
            payload["return_related_questions"] = True

        try:
            response = await self._http.post(
                CHAT_ENDPOINT, json=payload, headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"Chat API error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


def format_search_results(data: dict) -> str:
    """Format search API response into readable string."""
    if "error" in data:
        return data["error"]

    results = data.get("results", [])
    if not results:
        return "No search results found."

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. {result.get('title', 'No title')}")
        formatted.append(f"   URL: {result.get('url', 'No URL')}")
        if snippet := result.get("snippet"):
            formatted.append(f"   {snippet}")
        formatted.append("")
    return "\n".join(formatted)


def format_chat_response(data: dict) -> str:
    """Format chat completion response with citations."""
    if "error" in data:
        return data["error"]

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    output = [content]

    if search_results := data.get("search_results"):
        output.append("\n\nSources:")
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            output.append(f"{i}. [{title}]({url})")

    if images := data.get("images"):
        output.append("\n\nRelated Images:")
        for img_url in images[:5]:
            output.append(f"- {img_url}")

    if related := data.get("related_questions"):
        output.append("\n\nRelated Questions:")
        for question in related:
            output.append(f"- {question}")

    return "\n".join(output)
