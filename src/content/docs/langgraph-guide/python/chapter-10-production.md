---
title: "Chapter 10 — Production & Troubleshooting"
description: "Docker, CLI configuration, remote execution via the SDK, and fixes for the most common LangGraph errors."
framework: langgraph
language: python
sidebar:
  label: "10 · Production & troubleshooting"
  order: 10
---

# Chapter 10 — Production & Troubleshooting

**What you'll learn:** the smallest viable deployment path — a Docker image, a `langgraph.json` CLI config, and calling a deployed graph via the LangGraph SDK. Plus troubleshooting for the five errors you'll hit most often.

**Time:** ~20 minutes.

> For the full deployment playbook (Kubernetes, cost optimization, disaster recovery, observability), continue to the [Production Guide](/langgraph-guide/python/langgraph_production_guide/) after this chapter.

## Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start LangGraph server
CMD ["langgraph", "run", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t my-agent:v1 .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  my-agent:v1
```

### CLI Configuration

```json
{
  "langgraph.json": {
    "dependencies": [
      "langchain_anthropic",
      "langchain_tavily",
      "./agents"
    ],
    "graphs": {
      "main_agent": "./agents.py:graph",
      "research_agent": "./agents.py:research_graph"
    },
    "env": "./.env",
    "python_version": "3.11"
  }
}
```

### Remote Execution via SDK

```python
from langgraph_sdk import get_client
import asyncio

async def main():
    client = get_client(url="https://my-deployment.langraph.app")
    
    # List available assistants (from langgraph.json graphs)
    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]
    
    # Create conversation thread
    thread = await client.threads.create()
    
    # Stream execution
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input={"query": "Research AI trends"}
    ):
        if chunk.event == "messages/partial":
            print(chunk.data[0]["content"], end="", flush=True)
    
    # Get final state
    final_state = await client.threads.get_state(thread["thread_id"])
    print(f"\nFinal: {final_state}")

asyncio.run(main())
```

---

## Common Patterns Summary

| Pattern | Use Case | Key Idea |
|---------|----------|----------|
| **Linear** | Simple pipelines | Node A → B → C → END |
| **Conditional** | Decision trees | Routes based on state |
| **Looping** | Iterations | Self-referencing edges with exit condition |
| **Supervisor** | Multi-agent | Central router to specialists |
| **Parallel** | Concurrent work | Fan-out with Send, fan-in with collection |
| **ReAct** | Autonomous agent | Reason → Action → Observe loop |
| **Tree-of-Thoughts** | Complex reasoning | Multiple parallel thought paths |
| **Reflection** | Quality improvement | Self-critique → Refine loop |
| **Interrupt** | Human approval | Pause, wait, resume with Command |
| **Caching** | Performance | Store expensive results |


---

## Troubleshooting

### Issue: "Checkpointer must be provided for interrupts"

**Cause**: Trying to use `interrupt()` without a checkpointer  
**Fix**: Always compile with a checkpointer when using interrupts:

```python
graph = builder.compile(checkpointer=InMemorySaver())
```

### Issue: State not persisting across invocations

**Cause**: Missing `thread_id` in config  
**Fix**: Always provide consistent `thread_id`:


```python
config = {"configurable": {"thread_id": "unique-id"}}
result = graph.invoke(input, config=config)  # Same config each time
```


### Issue: Reducer functions not working

**Cause**: Not using `Annotated` with reducer function  
**Fix**: Proper state schema:

```python
# Wrong
class State(TypedDict):
    messages: list

# Correct
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### Issue: Tools not being called

**Cause**: Model not properly bound to tools  
**Fix**: Use `.bind_tools()`:

```python
model_with_tools = model.bind_tools(tools)
response = model_with_tools.invoke(messages)  # Works
```

### Issue: Infinite loops

**Cause**: Conditional edge always returns to same node  
**Fix**: Add iteration counter or state check:

```python
def should_continue(state) -> str:
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    return "process"
```

---

## Performance Tips

1. **Use async when possible**: `ainvoke()` and `astream()` for I/O-bound tasks
2. **Batch processing**: `graph.batch()` for multiple inputs
3. **Streaming**: Use `stream_mode="updates"` to reduce data transfer
4. **Checkpointer selection**: PostgreSQL > SQLite > In-Memory based on scale
5. **Cache expensive operations**: Store results in long-term Store
6. **Limit iterations**: Always set `MAX_ITERATIONS` to prevent runaway loops


---

## Next steps

You've finished the Zero → Hero path. 🎉 Where to go from here:

- **Build something real** — pick a recipe from the [Recipes collection](/langgraph-guide/python/langgraph_recipes/) (RAG, support router, research agent, doc pipeline, long-term memory chat).
- **Ship it** — read the full [Production Guide](/langgraph-guide/python/langgraph_production_guide/) covering Kubernetes, observability, cost tracking, and disaster recovery.
- **Scale it** — see [Performance Optimization](/langgraph-guide/python/langgraph_performance_optimization/) and [Observability](/langgraph-guide/python/langgraph_observability_python/).
- **Stream it** — the [FastAPI streaming server example](/langgraph-guide/python/langgraph_streaming_server_fastapi/) shows token-level SSE from a compiled graph.

Good luck, and welcome to durable, stateful agent systems.
