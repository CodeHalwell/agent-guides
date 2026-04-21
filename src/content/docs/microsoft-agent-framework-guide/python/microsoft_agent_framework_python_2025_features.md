---
title: "Microsoft Agent Framework (Python) — 2025 / 2026 Features"
description: "Verified feature set in agent-framework-core 1.1.0 and beta sub-packages (a2a, declarative, copilotstudio, observability) as of April 2026."
framework: microsoft-agent-framework
language: python
---

# 2025 / 2026 Features — Python

> **Errata (April 2026).** An earlier draft of this page documented `from agent_framework.a2a import A2AProtocolAdapter, A2AClient, A2AMessage` and `from agent_framework.graphs import AgentGraph, GraphNode, CheckpointConfig, HITLConfig` — **none of those classes exist**. This page has been rewritten after direct introspection of the installed packages.
>
> For the real graph/workflow API and A2A integration, see the dedicated pages:
> - [Workflows & Declarative Agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/)
> - [A2A Protocol](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/)

## Package status (April 2026)

| Package | Version | Status |
|---|---|---|
| `agent-framework-core` | 1.1.0 | Stable |
| `agent-framework-foundry` | 1.1.0 | Stable |
| `agent-framework-openai` | 1.1.0 | Stable |
| `agent-framework-a2a` | 1.0.0b260421 | Beta |
| `agent-framework-declarative` | 1.0.0b260421 | Beta |
| `agent-framework-copilotstudio` | 1.0.0b260421 | Beta |
| `agent-framework-mem0` | 1.0.0b260421 | Beta |
| `agent-framework-anthropic` | 1.0.0b260421 | Beta |
| `agent-framework-bedrock` | 1.0.0b260421 | Beta |
| `agent-framework-azurefunctions` | 1.0.0b260421 | Beta |
| `agent-framework-durabletask` | 1.0.0b260421 | Beta |
| `agent-framework-orchestrations` | 1.0.0b260421 | Beta |
| `agent-framework-redis` | 1.0.0b260421 | Beta |

The `agent-framework` meta package (1.1.0) installs core + provider packages in one go. When you're ready to prune dependencies, install the sub-packages directly (e.g. `pip install --pre agent-framework-foundry agent-framework-mem0`).

## Agent2Agent (A2A) protocol

`agent-framework-a2a` exposes exactly one class — `A2AAgent` — which wraps a remote A2A-compliant agent as a local framework `Agent`. Full detail on the [A2A protocol page](/microsoft-agent-framework-guide/microsoft_agent_framework_a2a_protocol/).

```python
from agent_framework.a2a import A2AAgent

remote = A2AAgent(url="https://analyst.example.com/a2a", name="Analyst")
response = await remote.run("Summarise Q3")
```

## Workflows (graph-based orchestration)

The real workflow API is `WorkflowBuilder` + `Executor`/`FunctionExecutor` + `Edge` groups, all exported from top-level `agent_framework`. See [Workflows & Declarative Agents](/microsoft-agent-framework-guide/microsoft_agent_framework_graphs_declarative/) for the full reference. One-glance example:

```python
from agent_framework import WorkflowBuilder, FunctionExecutor

research = FunctionExecutor(lambda q: {"data": q}, id="research")
write    = FunctionExecutor(lambda d: f"report: {d['data']}", id="write")

workflow = (
    WorkflowBuilder(start_executor=research, name="ContentPipeline")
    .add_edge(research, write)
    .build()
)
```

Workflows support **checkpointing** via `FileCheckpointStorage` / `InMemoryCheckpointStorage`, **fan-in / fan-out / switch-case** routing, and **exposing a workflow as an agent** via `WorkflowAgent`.

## Declarative agents & workflows (YAML)

`agent-framework-declarative` provides `AgentFactory` (single agent from YAML) and `WorkflowFactory` (multi-agent workflow from YAML). Actions are a Power-Fx-expression dialect — `SetVariable`, `InvokeAzureAgent`, `InvokeFunctionTool`, `If`, `Foreach`, `RepeatUntil`, `Question`, `Confirmation`, `SendActivity`, etc. Full reference: [Microsoft Learn — Declarative Workflows](https://learn.microsoft.com/agent-framework/workflows/declarative/).

```python
from agent_framework.declarative import AgentFactory

agent = AgentFactory.create_from_yaml_path("agent.yaml")
result = await agent.run("Hello")
```

## Middleware & hooks

Middleware is a first-class concept in `agent-framework-core`. Three decorators — `@chat_middleware`, `@agent_middleware`, `@function_middleware` — and three base classes — `ChatMiddleware`, `AgentMiddleware`, `FunctionMiddleware` — let you wrap the corresponding call type:

```python
from agent_framework import Agent, chat_middleware, ChatContext
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import DefaultAzureCredential

@chat_middleware
async def logging_middleware(context: ChatContext, next):
    print(f"before: {len(context.messages)} messages")
    await next(context)
    print(f"after: {context.result}")

agent = Agent(
    client=FoundryChatClient(credential=DefaultAzureCredential(), model="gpt-4o"),
    instructions="You are a helpful assistant.",
    middleware=[logging_middleware],  # must be a list in 2026 releases
)
```

Breaking changes in the 2026 beta line (verified against release notes):

- `display_name` parameter removed from agents.
- `context_providers` (plural list) renamed to `context_provider` (singular; only one provider allowed).
- `middleware` now requires a list — no longer accepts a single instance.
- `AgentRunResponse` / `AgentRunResponseUpdate` renamed to `AgentResponse` / `AgentResponseUpdate`.
- `AggregateContextProvider` removed — build your own aggregator from examples.

## OpenTelemetry observability

`agent-framework-core` emits OpenTelemetry traces and metrics out of the box. Wire it up with the standard OTel SDK:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry import trace

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Framework code from here on emits agent_framework.* spans
```

For Azure Monitor, use `azure.monitor.opentelemetry.configure_azure_monitor(connection_string=...)`.

## Content safety

Content safety is **not** a framework-specific wrapper class. Call Azure AI Content Safety directly from a middleware:

```python
from agent_framework import chat_middleware, ChatContext
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

cs_client = ContentSafetyClient("https://<region>.cognitiveservices.azure.com/",
                                AzureKeyCredential("<key>"))

@chat_middleware
async def safety_guard(context: ChatContext, next):
    last_user = next((m for m in reversed(context.messages) if m.role == "user"), None)
    if last_user:
        verdict = await cs_client.analyze_text(text=last_user.content)
        if verdict.categories_analysis[0].severity > 2:
            raise ValueError("Blocked by content safety")
    await next(context)
```

## Async patterns

The framework is async-first:

- Agents expose `await agent.run(...)` and `async for update in agent.run(..., stream=True): ...`.
- Chat clients implement the `SupportsChatGetResponse` protocol — `await client.get_response(...)`.
- Checkpoint storage (`FileCheckpointStorage`, `InMemoryCheckpointStorage`) exposes awaitable `save` / `load` / `get_latest`.
- `A2AAgent` uses `httpx.AsyncClient` under the hood; pass your own if you need custom TLS / retry / auth.

## Production patterns

- **Package minimisation.** Pin only the sub-packages you need: `pip install --pre agent-framework-foundry agent-framework-redis`.
- **Session persistence.** Use `agent.create_session()` with the Redis, Cosmos, or mem0 providers depending on your backend.
- **Checkpointing long-running workflows.** Use `FileCheckpointStorage` for local, or wrap a `CheckpointStorage`-compatible backend for cloud storage.
- **Durable agents on Azure Functions.** `agent-framework-azurefunctions` + `agent-framework-durabletask` expose `DurableAIAgent`, `DurableAIAgentClient`, and `AgentFunctionApp` for long-running orchestrations.
- **Multiple LLM providers.** Swap Foundry, OpenAI, Anthropic, Bedrock, Ollama, and local Foundry via the chat-client of the same interface (`SupportsChatGetResponse`).

## Further reading

- [Microsoft Learn — Agent Framework](https://learn.microsoft.com/agent-framework/)
- [Migration from Semantic Kernel](https://learn.microsoft.com/agent-framework/migration-guide/from-semantic-kernel/)
- [Source (microsoft/agent-framework)](https://github.com/microsoft/agent-framework)
