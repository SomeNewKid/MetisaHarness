# Metisa Harness

Metisa Harness is a small Python framework for learning how AI agents work.
It follows the division of responsibilities in the OpenAI Agents SDK.
`AgentDefinition` plays the role of `Agent`, describing the model, instructions,
tools, and optional skills. `AgentHarness` plays the role of `Runner`, executing
the model and tool loop for one agent.

The sister project, `MetisaSandbox`, uses `metisa.toml` to build a Docker-based
sandbox for agent workloads. The harness is intended to run inside that sandbox
eventually. For now, the projects are developed separately so the harness can be
run directly during development. Shared code lives in `metisa_common`.

## How a run works

1. `AgentHarness.run()` registers synchronous or asynchronous Python tools and
   discovers tools from any connected MCP clients supplied in `AgentDefinition`.
   It also exposes agents listed in `sub_agents` as tools. Each call starts a
   child run and returns its final output to the parent as a tool result.
   When a skills directory is configured, the harness discovers its skills and
   registers the tools for using them if it finds any.
2. The harness sends the instructions, available tools, and conversation to the
   model. The instructions include the names and descriptions of discovered
   skills. By default, each model call receives the full history of the current run.
3. When the model requests tools, the harness handles the calls in order. It
   validates each call's arguments, executes the tool, checks the result against
   any declared output schema, and sends the results back on the next turn.
4. When the model returns a final answer, the harness returns it with the
   conversation history in `RunResult`. Set `output_type` to a Pydantic model to
   get validated structured output. If the model never produces a final answer,
   the run raises an error after `max_turns` (10 by default).

Guardrail plugins can inspect, transform, or reject data before and after a run,
model call, or tool call. An optional async `event_handler` observes run, model,
and tool lifecycle events. Callers can supply a custom `llm` or `context_builder`.

## Code layout

- [`src/metisa_harness`](src/metisa_harness) contains the agent definition,
  execution loop, model adapter, tool dispatch, skills, guardrails, and lifecycle
  events.
- [`src/sample_agent`](src/sample_agent) contains the Windows CLI, example tools,
  robot inspection skills, guardrails, and in-memory MCP server.
- [`src/metisa_common`](src/metisa_common) contains shared specification parsing
  and validation, tool schema helpers, secrets handling, screening, and service
  clients. The harness uses its tool schema helper, and the sample CLI uses its
  specification loader.

## Development setup

Requires Python 3.11 or newer. Check the local Python installation path in
`scripts/setup-dev.ps1`, then run:

```powershell
.\scripts\setup-dev.ps1
```

This creates `.venv` and installs the project in editable mode with development
tools (Ruff, Pyright, and pytest).

## Run the sample

Start Ollama on its default local port (`11434`) and install `granite4.1:3b` for
the local run. Set `OPENAI_API_KEY` in your environment for the OpenAI run.
Then run the Windows CLI from the repository root.

```powershell
.\.venv\Scripts\python.exe -m sample_agent "What is the time?"
```

The CLI runs the same prompt with `gpt-4o` and `granite4.1:3b`. Try
`Look up product code "chair".` to exercise the product tool and its guardrails,
or `How many company values do we have?` to exercise the MCP document tools,
which return example handbook excerpts. The CLI connects to its in-memory MCP
server in Python. The `[mcp_server]` table in the sample `metisa.toml` is for
`MetisaSandbox`. With the sample's telemetry capability enabled, the CLI writes
traces to `.runs/<run-id>/.logs/telemetry/traces.json`.

Run the [robot inspection skills](src/sample_agent/skills) with these prompts.

```powershell
.\.venv\Scripts\python.exe -m sample_agent "Can the robot named Bolt enter the race?"
.\.venv\Scripts\python.exe -m sample_agent "Can the robot named Alice be placed in storage?"
```

Both skills use the same battery and motor-temperature tools. The race skill
loads a reference with the race limits. The storage skill loads a checklist and,
if both measurements pass, runs `storage_available.star` to check for a bay.
The sample data gives Bolt passing race measurements and Alice passing storage
measurements with a bay available. The CLI prints lifecycle events so you can
see the skill, document, and script tool calls. With the current configuration,
the Alice question is also the default when you omit the prompt.

## Use the harness from Python

```python
import asyncio

from metisa_harness import AgentDefinition, AgentHarness


async def main() -> None:
    agent = AgentDefinition(
        name="sample_agent",
        description="Answers users' questions.",
        model="gpt-4o",
        instructions="Answer the user's questions concisely.",
    )
    result = await AgentHarness.run(
        agent,
        run_identifier="example-run",
        input="What is an AI agent harness?",
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

To add a local tool, pass it in `tools`. Give the function type hints and attach
a JSON output schema with `metisa_common.mcp_helper.with_output_schema`. For MCP
tools, open the clients in the caller and pass them in `mcp_clients`.

The default model resolver checks Ollama before using the OpenAI mappings for
`gpt4o` and `gpt-4o`. If Ollama is unreachable, model resolution fails even for
an OpenAI alias. Pass `llm=` to use a model client directly.

## Add a skill

Set `skills_directory=Path("path/to/skills")` on `AgentDefinition`. Put each
skill in its own immediate child directory with a `SKILL.md` file. Include `name`
and `description` in the YAML front matter and the procedure in the Markdown
body. Refer to any supporting files by paths relative to the skill directory.
The sample places supporting documents in `references/` and scripts in `scripts/`.

The harness uses **progressive disclosure**. It gives the model the skill names
and descriptions first. The model can then request the full procedure and any
files it needs through the skill tools.

| Tool | Arguments | Result |
| --- | --- | --- |
| `load_skill` | `skill_name` | The body of `SKILL.md` as `content` |
| `get_skill_document` | `skill_name`, `document_path` | The document text as `content` |
| `execute_skill_script` | `skill_name`, `script_path`, `script_parameters` | The script return value as a string in `result` |

For the storage example, the script path is `scripts/storage_available.star` and
`script_parameters` is `{"robot_name": "Alice"}`. The local development runner
loads the script with Python `exec()` and then calls `main` with those named
arguments. Use this runner only with trusted, Python-compatible sample scripts.
The hardened Starlark VM is provided by `MetisaSandbox` for the future integration.

## Development checks

```powershell
.\scripts\check.ps1
```

This formats the code, runs lint and type checks, and executes the tests.
