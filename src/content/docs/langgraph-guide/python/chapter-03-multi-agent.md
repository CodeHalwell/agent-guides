---
title: "Chapter 3 — Multi-Agent Systems"
description: "Supervisor routing, parallel fan-out/fan-in, and hand-off patterns for coordinating multiple specialist agents."
framework: langgraph
language: python
sidebar:
  label: "3 · Multi-agent systems"
  order: 3
---

# Chapter 3 — Multi-Agent Systems

**What you'll learn:** three canonical multi-agent topologies — a supervisor routing to specialists, parallel workers with fan-out/fan-in, and direct hand-off between agents.

**Time:** ~25 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

## Multi-Agent Systems

### Example 1: Supervisor Pattern

One coordinator agent routing to specialists:


```python
from langchain_core.messages import BaseMessage
# Note: AgentExecutor and create_tool_calling_agent require `pip install langchain langchain-anthropic`
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langchain_core.tools import tool
from typing import List

# Define specialized agents' tools
@tool
def research_tool(query: str) -> str:
    """Search the web for information."""
    return f"Research results for: {query}"

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate math expressions."""
    # In a real scenario, use a safe evaluation library
    return str(eval(expression))

# Helper function to create a specialist agent
def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Create agent runner function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [BaseMessage(type="human", content=result["output"], name=name)]}

# Create specialized agents
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
research_agent = create_agent(model, [research_tool], "You are a research specialist. Find accurate information.")
math_agent = create_agent(model, [calculator_tool], "You are a math specialist. Solve problems step-by-step.")

# Supervisor state
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# Supervisor logic
def supervisor_node(state: SupervisorState) -> dict:
    """Analyze request and pick best agent."""
    last_message = state["messages"][-1]
    
    # If the last message is from an agent, the supervisor can decide to end the process
    if hasattr(last_message, 'name'):
        return {"next": "END"}

    prompt = f"""You manage two specialist agents:
- research_agent: For web searches, fact-finding, current info
- math_agent: For calculations and equations

Request: {last_message.content}

Which agent should handle this? Reply with ONLY the agent name or FINISH."""
    
    response = model.invoke(prompt)
    next_agent = response.content.strip()
    
    return {"next": next_agent}

# Build supervisor graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research_agent", lambda state: agent_node(state, research_agent, "research_agent"))
builder.add_node("math_agent", lambda state: agent_node(state, math_agent, "math_agent"))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "research_agent": "research_agent",
        "math_agent": "math_agent",
        "FINISH": END,
    }
)

# Agents return to supervisor
builder.add_edge("research_agent", "supervisor")
builder.add_edge("math_agent", "supervisor")

supervisor_graph = builder.compile(checkpointer=InMemorySaver())

# Test it
config = {"configurable": {"thread_id": "supervisor-test"}}

result = supervisor_graph.invoke(
    {"messages": [{"role": "user", "content": "Research AI trends and calculate 25% of 1000"}]},
    config=config
)

print("Final response:", result["messages"][-1].content)
```


### Example 2: Parallel Worker Pattern

Fan-out to multiple workers, collect results:


```python
from langgraph.types import Send

class WorkflowState(TypedDict):
    tasks: list[dict]
    results: Annotated[dict, lambda x, y: {**x, **y}]  # Merge dicts

def split_tasks(state: WorkflowState) -> list[Send]:
    """Create parallel work for each task."""
    return [
        Send(
            "worker",
            {
                "task_id": task["id"],
                "task_data": task["data"]
            }
        )
        for task in state["tasks"]
    ]

def worker_node(state: WorkflowState) -> dict:
    """Process one task."""
    # Simulate work
    result = f"Processed: {state['task_data']}"
    return {"results": {state["task_id"]: result}}

def collect_results(state: WorkflowState) -> dict:
    """Aggregate all results."""
    summary = f"Completed {len(state['results'])} tasks"
    return {"results": {"summary": summary}}

# Build parallel graph
builder = StateGraph(WorkflowState)
builder.add_node("split", split_tasks)
builder.add_node("worker", worker_node)
builder.add_node("collect", collect_results)

# Fan-out: split → multiple workers
builder.add_conditional_edges(
    START,
    lambda _: "split"
)
builder.add_conditional_edges(
    "split",
    lambda _: ["worker"],  # All Send objects go to worker
    ["worker"]
)

# Fan-in: collect all results
builder.add_edge("worker", "collect")
builder.add_edge("collect", END)

parallel_graph = builder.compile()

# Test
result = parallel_graph.invoke({
    "tasks": [
        {"id": "task-1", "data": "data-a"},
        {"id": "task-2", "data": "data-b"},
        {"id": "task-3", "data": "data-c"}
    ]
})

print("Results:", result["results"])
# Output: {'task-1': 'Processed: data-a', 'task-2': 'Processed: data-b', ...}
```


### Example 3: Handoff Pattern

Agents handing off to each other mid-conversation:


```python
class HandoffState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    handoff_reason: str

def agent_a(state: HandoffState) -> dict:
    """First agent - handles initial request."""
    last_message = state["messages"][-1].content
    
    # Check if should handoff
    if "transfer" in last_message.lower():
        return {
            "current_agent": "agent_b",
            "handoff_reason": "User requested transfer",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Transferring to agent B..."
                }
            ]
        }
    
    # Normal response
    response = f"Agent A responds to: {last_message}"
    return {
        "current_agent": "agent_a",
        "messages": [{"role": "assistant", "content": response}]
    }

def agent_b(state: HandoffState) -> dict:
    """Second agent - takes over."""
    last_message = state["messages"][-1].content
    response = f"Agent B (now handling): {last_message}"
    return {
        "current_agent": "agent_b",
        "messages": [{"role": "assistant", "content": response}]
    }

def route_agent(state: HandoffState) -> str:
    """Route to current agent."""
    agent = state.get("current_agent", "agent_a")
    return agent

# Build handoff graph
builder = StateGraph(HandoffState)
builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)

builder.add_edge(START, "agent_a")
builder.add_conditional_edges(
    "agent_a",
    lambda state: "agent_b" if state.get("current_agent") == "agent_b" else "agent_a"
)
builder.add_edge("agent_b", END)

handoff_graph = builder.compile(checkpointer=InMemorySaver())

# Test handoff
config = {"configurable": {"thread_id": "handoff-test"}}

result = handoff_graph.invoke(
    {"messages": [{"role": "user", "content": "Help me"}], "current_agent": "agent_a"},
    config=config
)
print("Step 1:", result["messages"][-1].content)

result = handoff_graph.invoke(
    {"messages": [{"role": "user", "content": "Transfer me to another agent"}]},
    config=config
)
print("Step 2:", result["messages"][-1].content)
print("Current agent:", result["current_agent"])
```


