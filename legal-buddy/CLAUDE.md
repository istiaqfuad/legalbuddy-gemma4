## Project Setup

This project uses **uv** for dependency management and task execution.

### Common Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Run tests
uv run pytest

# Add a dependency
uv add <package>

# Add a development dependency
uv add --dev <package>
```

## Development Guidelines

* Prefer `uv run` over directly invoking Python or installed tools.
* Follow existing code style and project conventions.
* Keep changes minimal and focused on the requested task.
* Run relevant tests before completing significant changes.

## Environment Variables

**Do not read or inspect `.env` files.**

If you need to understand available environment variables, refer to `.env.example` instead. Treat that file as the source of truth for configuration documentation.

Assume sensitive values may exist in `.env` and should not be accessed unless explicitly provided by the user.

## Before Making Changes

1. Read relevant source files.
2. Understand the existing architecture and patterns.
3. Check for existing tests covering the affected functionality.
4. Prefer modifying existing code over introducing new abstractions.

## Output Expectations

* Explain significant code changes briefly.
* Highlight any assumptions made.
* Mention any tests that were run or should be run.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
