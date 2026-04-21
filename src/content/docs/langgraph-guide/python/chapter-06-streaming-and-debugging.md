---
title: "Chapter 6 — Streaming & Debugging"
description: "Stream values/updates/debug events, visualize graphs, and adopt the type-safe v2 API introduced in v1.1.x."
framework: langgraph
language: python
sidebar:
  label: "6 · Streaming & debugging"
  order: 6
---

# Chapter 6 — Streaming & Debugging

**What you'll learn:** how to see what your graph is doing at runtime. Stream events as they happen, render the graph as a Mermaid diagram, inspect and modify state at any checkpoint, and adopt the type-safe v2 streaming/invoke API added in v1.1.

**Time:** ~15 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

## Debugging & Visualization

### Graph Visualization

```python
from IPython.display import Image, display

# Get Mermaid diagram
diagram = graph.get_graph().draw_mermaid()
print(diagram)

# Display in Jupyter/Colab
display(Image(graph.get_graph().draw_mermaid_png()))

# ASCII art
print(graph.get_graph().draw_ascii())
```

Example output:

```
    ┌─────────────────────┐
    │      START          │
    └────────────┬────────┘
                 │
    ┌────────────▼────────────┐
    │     fetch_context       │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │     call_model          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │      save_chat          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │        END              │
    └─────────────────────────┘
```

### Streaming for Debugging


```python
# Debug mode shows node execution
config = {"configurable": {"thread_id": "debug-1"}}

for event in graph.stream(
    {"query": "test"},
    config=config,
    stream_mode="debug"
):
    print(f"Event: {event}")

# Output:
# {'type': 'task_start', 'timestamp': '...', 'step': 0, 'node': 'fetch_context'}
# {'type': 'task_end', 'timestamp': '...', 'step': 0, 'node': 'fetch_context', 'result': {...}}
# {'type': 'task_start', 'timestamp': '...', 'step': 1, 'node': 'call_model'}
# ...
```


## Type-Safe v2 API (v1.1.x)

### v2 Streaming

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    result: str

builder = StateGraph(State)
# ... add nodes and edges ...
graph = builder.compile()

# v2 streaming: opt-in with version="v2"
async for part in graph.astream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    version="v2",  # Enables type-safe StreamPart output
):
    # part is a StreamPart with .type, .ns, and .data
    print(f"Type: {part.type}, Data: {part.data}")
```

### v2 Invoke

```python
# v2 invoke returns a GraphOutput instead of a dict
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    version="v2",
)

# GraphOutput has .value (final state) and .interrupts (any Human-in-the-Loop interrupts)
print(result.value)       # Final state dict
print(result.interrupts)  # List of interrupt points (if any)
```

### Pydantic/Dataclass Auto-Coercion

v1.1.x automatically coerces input dictionaries to the graph's state type on `invoke()`:

```python
from pydantic import BaseModel

class MyState(BaseModel):
    query: str
    result: str = ""

builder = StateGraph(MyState)
# ... graph setup ...
graph = builder.compile()

# Pass dict directly — auto-coerced to MyState
result = await graph.ainvoke({"query": "What is LangGraph?"})
print(type(result))  # MyState
```

### Getting State at Any Point


```python
# After partial execution
config = {"configurable": {"thread_id": "user-1"}}

# Start but intercept in middle
for event in graph.stream({"query": "test"}, config=config):
    pass

# Get state snapshot
state = graph.get_state(config)
print(f"Next node to run: {state.next}")
print(f"Current values: {state.values}")
print(f"Metadata: {state.metadata}")

# Modify state
graph.update_state(
    config,
    {"messages": [{"role": "system", "content": "Updated system prompt"}]}
)

# Continue from modified state
result = graph.invoke({"query": "continue"}, config=config)
```


### Checkpoint Inspection


```python
# List all checkpoints for a thread
config = {"configurable": {"thread_id": "user-1"}}

history = list(graph.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

for i, snapshot in enumerate(history):
    cp_id = snapshot.config['configurable']['checkpoint_id']
    next_node = snapshot.next
    print(f"\nCheckpoint {i}: {cp_id}")
    print(f"  Next node(s): {next_node}")
    print(f"  State keys: {list(snapshot.values.keys())}")
```


### Batch Debugging


```python
# Process multiple and collect issues
inputs = [
    {"query": "Query 1"},
    {"query": "Query 2"},
    {"query": "Query 3"}
]

configs = [
    {"configurable": {"thread_id": f"batch-{i}"}}
    for i in range(len(inputs))
]

results = []
errors = []

for inp, cfg in zip(inputs, configs):
    try:
        result = graph.invoke(inp, config=cfg)
        results.append(result)
    except Exception as e:
        errors.append((cfg["configurable"]["thread_id"], str(e)))

print(f"Successful: {len(results)}/{len(inputs)}")
print(f"Failed: {len(errors)}")
for thread_id, error in errors:
    print(f"  {thread_id}: {error}")
```

