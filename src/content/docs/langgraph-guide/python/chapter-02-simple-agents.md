---
title: "Chapter 2 — Your First Agent"
description: "Build linear pipelines, conditional routers, looping agents, and streaming outputs — four concrete patterns that cover most single-agent workloads."
framework: langgraph
language: python
sidebar:
  label: "2 · Your first agent"
  order: 2
---

# Chapter 2 — Your First Agent

**What you'll learn:** four concrete agent patterns you can copy-paste and adapt — a linear chat pipeline, conditional routing, looping with a safeguard counter, and streaming execution.

**Time:** ~20 minutes.

> Prereqs: [Chapter 1 — Setup & Core Concepts](/langgraph-guide/python/chapter-01-setup-and-core-concepts/).

## Simple Agents

### Example 1: Linear Chat Pipeline

A basic chatbot with no branching:


```python
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str

# Construct the model ONCE at module scope and reuse it across every node
# invocation — creating a ChatAnthropic client per call is unnecessarily
# expensive in production. Nodes should stay focused on state transforms.
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def fetch_user_context(state: State) -> dict:
    """Load user info from database."""
    # Simulate DB lookup
    return {"user_name": "Alice"}

def call_model(state: State) -> dict:
    """Call LLM with messages."""
    system_prompt = f"You're helping {state['user_name']}. Be concise."
    response = model.invoke(state["messages"], system_prompt=system_prompt)
    return {"messages": [response]}

def save_conversation(state: State) -> dict:
    """Persist messages to database."""
    # Save state["messages"] to DB
    return {}

# Build the graph
builder = StateGraph(State)
builder.add_node("fetch_context", fetch_user_context)
builder.add_node("model", call_model)
builder.add_node("save", save_conversation)

builder.add_edge(START, "fetch_context")
builder.add_edge("fetch_context", "model")
builder.add_edge("model", "save")
builder.add_edge("save", END)

# Compile with persistence
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

# Use it
config = {"configurable": {"thread_id": "chat-session-1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    config=config
)

# Continue in same thread - context preserved
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What did you say before?"}]},
    config=config
)
```


### Example 2: Conditional Routing

Route based on message type:

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    query_type: str
    result: str

# Reuse a single model instance across all node calls.
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def classify_query(state: State) -> dict:
    """Determine query type."""
    query = state["query"].lower()

    if any(word in query for word in ["search", "find", "lookup"]):
        return {"query_type": "search"}
    elif any(word in query for word in ["calculate", "math", "solve"]):
        return {"query_type": "math"}
    else:
        return {"query_type": "general"}

def search_web(state: State) -> dict:
    """Handle search queries."""
    # Call search API
    result = f"Search results for: {state['query']}"
    return {"result": result}

def solve_math(state: State) -> dict:
    """Handle math queries."""
    result = f"Math answer for: {state['query']}"
    return {"result": result}

def general_response(state: State) -> dict:
    """Handle general queries."""
    response = model.invoke(state["query"])
    return {"result": response.content}

# Build graph with conditional routing
builder = StateGraph(State)
builder.add_node("classify", classify_query)
builder.add_node("search", search_web)
builder.add_node("math", solve_math)
builder.add_node("general", general_response)

# Route based on classification
def route_to_handler(state: State) -> str:
    return state["query_type"]

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_to_handler,
    {
        "search": "search",
        "math": "math",
        "general": "general"
    }
)

# All handlers lead to END
for handler in ["search", "math", "general"]:
    builder.add_edge(handler, END)

graph = builder.compile()

# Test it
result = graph.invoke({"query": "What's the population of Tokyo?"})
print(result["result"])  # Routes to search

result = graph.invoke({"query": "Calculate 15% of 2000"})
print(result["result"])  # Routes to math
```

### Example 3: Looping with Counter

Agent that can loop (with limits):


```python
class LoopState(TypedDict):
    iteration: int
    data: str
    final_result: str

def process_step(state: LoopState) -> dict:
    """Do one iteration of processing."""
    processed = state["data"] + f" [step-{state['iteration']}]"
    return {
        "data": processed,
        "iteration": state["iteration"] + 1
    }

def should_continue(state: LoopState) -> str:
    """Decide whether to loop or finish."""
    if state["iteration"] >= 3:
        return "finish"
    return "continue"

def finalize(state: LoopState) -> dict:
    """Final processing."""
    return {"final_result": state["data"]}

builder = StateGraph(LoopState)
builder.add_node("process", process_step)
builder.add_node("finalize", finalize)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back to self
        "finish": "finalize"
    }
)
builder.add_edge("finalize", END)

graph = builder.compile()

# Looping with safeguard
config = {"configurable": {"thread_id": "loop-test"}}
result = graph.invoke(
    {"iteration": 0, "data": "start"},
    config=config
)
print(result)
# Output: {'iteration': 3, 'data': 'start [step-0] [step-1] [step-2]', 'final_result': '...'}
```


### Example 4: Streaming Output

See the graph execute step-by-step:


```python
# Different streaming modes
config = {"configurable": {"thread_id": "stream-test"}}

# Mode 1: "values" - full state after each step
print("=== Streaming Values ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="values"
):
    print(f"State: {event}\n")

# Mode 2: "updates" - only what changed
print("\n=== Streaming Updates ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="updates"
):
    for node_name, updates in event.items():
        print(f"{node_name} updated: {updates}\n")

# Mode 3: "debug" - node execution trace
print("\n=== Debug Mode ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="debug"
):
    print(f"Debug: {event}\n")
```


---
