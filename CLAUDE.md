# cassandra-perplexity-mcp

Perplexity Sonar API MCP server — web search and grounded AI answers.

## Architecture

- **Search endpoint** — raw results with titles, URLs, snippets
- **Chat completions** — AI-synthesized answers with sonar/sonar-pro/sonar-reasoning-pro
- **Recency filtering** — passthrough (hour/day/week/month/year) and computed date ranges (today/yesterday/last_week/last_month)

## Tools

| Tool | Model | Purpose |
|------|-------|---------|
| `search` | Search API | Raw source results (titles, URLs, snippets) |
| `ask` | sonar / sonar-pro | AI-synthesized answers with citations |
| `ask_reasoning` | sonar-reasoning-pro | Step-by-step reasoning with citations |

## Config

| Env Var | Required | Default |
|---------|----------|---------|
| `PERPLEXITY_API_KEY` | yes | — |
| `MCP_PORT` | no | `3004` |

## Dev

```bash
cd backend
uv sync
PERPLEXITY_API_KEY=... uv run cassandra-perplexity-mcp
```
