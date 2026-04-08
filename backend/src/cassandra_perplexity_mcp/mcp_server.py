"""FastMCP server for Cassandra Perplexity MCP — web search and grounded AI answers via Sonar API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from cassandra_mcp_auth import AclMiddleware
from cassandra_perplexity_mcp.auth import McpKeyAuthProvider, build_auth
from cassandra_perplexity_mcp.clients.perplexity import PerplexityClient
from cassandra_perplexity_mcp.config import Settings

logger = logging.getLogger(__name__)

SERVICE_ID = "perplexity-mcp"


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server with auth and all tools."""

    if not settings.perplexity_api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable is required. "
            "Get your API key from https://www.perplexity.ai/settings/api"
        )

    # Auth
    auth_provider = None
    mcp_key_provider = None
    if settings.auth_url and settings.auth_secret:
        if (
            settings.workos_client_id
            and settings.workos_authkit_domain
            and settings.base_url
        ):
            auth_provider, mcp_key_provider = build_auth(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
                base_url=settings.base_url,
                workos_client_id=settings.workos_client_id,
                workos_authkit_domain=settings.workos_authkit_domain,
            )
        else:
            mcp_key_provider = McpKeyAuthProvider(
                acl_url=settings.auth_url,
                acl_secret=settings.auth_secret,
                service_id=SERVICE_ID,
            )
            auth_provider = mcp_key_provider

    perplexity_client = PerplexityClient(api_key=settings.perplexity_api_key)

    # Set fallback so tools work even without lifespan (gateway embedding)
    from cassandra_perplexity_mcp.tools._helpers import set_fallback_client
    set_fallback_client(perplexity_client)

    @asynccontextmanager
    async def lifespan(server):
        yield {
            "perplexity_client": perplexity_client,
        }
        await perplexity_client.close()
        if mcp_key_provider is not None:
            mcp_key_provider.close()

    acl_mw = AclMiddleware(service_id=SERVICE_ID, acl_path=settings.auth_yaml_path)

    mcp_kwargs: dict = {
        "name": "Cassandra Perplexity",
        "instructions": (
            "# Cassandra Perplexity\n\n"
            "Web search and grounded AI answers powered by Perplexity Sonar API.\n\n"
            "## Tools\n\n"
            "- **search** — Find and evaluate sources yourself (URLs, titles, snippets)\n"
            "- **ask** — Get AI-synthesized answers with web-grounded citations\n"
            "- **ask_reasoning** — Get answers with explicit step-by-step reasoning\n\n"
            "All tools support recency filters and domain filtering."
        ),
        "lifespan": lifespan,
        "middleware": [acl_mw] if acl_mw._enabled else [],  # noqa: SLF001
    }
    if auth_provider:
        mcp_kwargs["auth"] = auth_provider

    mcp = FastMCP(**mcp_kwargs)

    # Health check
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request):  # noqa: ANN001, ARG001
        from starlette.responses import JSONResponse  # noqa: PLC0415

        return JSONResponse({"ok": True, "service": "cassandra-perplexity-mcp"})

    # Register tools
    from cassandra_perplexity_mcp.tools import register_all  # noqa: PLC0415

    register_all(mcp, settings)

    return mcp
