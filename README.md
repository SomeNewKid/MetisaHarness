# Metisa Harness

An evolving AI agent workbench for learning how agents work and experimenting
with their behavior. The implementation is deliberately small so that the
conversation loop, model requests, and tool execution remain easy to follow.

This is an unfinished learning project. Its current focus is tool use: giving a
model access to functions, executing its requests, and returning the results so
that it can answer the user.

## Current capabilities

- Define an agent with instructions, a model, local functions, and MCP servers.
- Use synchronous or asynchronous local tools and discover tools from MCP servers.
- Validate tool arguments and results against their JSON schemas.
- Run successive model turns and tool calls until a final answer or turn limit.
- Inspect the final answer and conversation history through `RunResult`.
- Use OpenAI or local Ollama models through Chat Completions, preferring an
  installed Ollama model when its name matches.

## Code layout

- `src/metisa_harness`: agent definitions, the execution loop, model selection,
  Chat Completions adaptation, and tool registration and invocation.
- `src/sample_agent`: a Windows CLI demonstrating the harness with a local
  date-and-time tool.
- `src/metisa_common`: utilities shared with the sister `MetisaSandbox` project,
  which creates Docker-based environments for agent workloads. The harness is
  intended to run within those environments as it develops.

## Development setup

Requires Python 3.11 or newer. Check the local Python installation path in
`scripts/setup-dev.ps1`, then run:

```powershell
.\scripts\setup-dev.ps1
```

This creates `.venv` and installs the project in editable mode with development
tools (Ruff, Pyright, and pytest).

## Run the sample

```powershell
.\.venv\Scripts\python.exe -m sample_agent "What is the time?"
```

The sample currently tries the same prompt with OpenAI and three configured
Ollama models, using the date-and-time tool. OpenAI requests require
`OPENAI_API_KEY`; local requests require a running Ollama server with the named
models installed. Adjust the model list in `src/sample_agent/cli.py` for your
experiments.

## Development checks

```powershell
.\scripts\check.ps1
```

This formats the code, runs lint and type checks, and executes the tests.
