---
title: "Chapter 4 — Tools"
description: "ToolNode, custom tool executors, and conditional tool usage — plus error handling, retries, and structured tool outputs."
framework: langgraph
language: python
sidebar:
  label: "4 · Tools"
  order: 4
---

# Chapter 4 — Tools

**What you'll learn:** how to plug external capabilities into your graph — the built-in `ToolNode`, writing your own tool executor for fine-grained control, and routing only through tools when the model requests them.

**Time:** ~20 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

## Tool Integration

### Example 1: Basic Tool Node

Using LangGraph's built-in `ToolNode`:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    prices = {"AAPL": 150.25, "GOOGL": 140.50}
    return f"{symbol}: ${prices.get(symbol, 'N/A')}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"

tools = [get_weather, get_stock_price, send_email]

# Create model with tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_results: list[str]

def agent_node(state: ToolState) -> dict:
    """Call model which may invoke tools."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph with tool handling
builder = StateGraph(ToolState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# tools_condition: Routes to "tools" if tool_calls exist, else END
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

# After tools, return to agent for next iteration
builder.add_edge("tools", "agent")

tool_graph = builder.compile()

# Use it
result = tool_graph.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in London and AAPL stock price?"}
    ]
})

print("Final response:", result["messages"][-1].content)
```

### Example 2: Custom Tool Executor

Handle tool execution yourself for more control:

```python
from langchain_core.messages import ToolMessage
import json

class CustomToolState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_errors: Annotated[list, lambda x, y: x + y]

def execute_tools(state: CustomToolState) -> dict:
    """Manually execute tool calls with error handling."""
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls"):
        return {}
    
    tool_results = []
    errors = []
    
    for tool_call in last_message.tool_calls:
        try:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            
            if tool_name == "get_weather":
                result = get_weather(args["city"])
            elif tool_name == "get_stock_price":
                result = get_stock_price(args["symbol"])
            else:
                result = "Tool not found"
            
            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )
        except Exception as e:
            errors.append(f"Tool {tool_name} failed: {str(e)}")
            tool_results.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {
        "messages": tool_results,
        "tool_errors": errors if errors else []
    }

# Build with custom tool executor
builder = StateGraph(CustomToolState)
builder.add_node("agent", agent_node)
builder.add_node("tools", execute_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    lambda state: "tools" if hasattr(state["messages"][-1], "tool_calls") else END,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "agent")

custom_tool_graph = builder.compile()
```

### Example 3: Conditional Tool Usage

Only use tools when needed:

```python
class ConditionalToolState(TypedDict):
    query: str
    use_tools: bool
    result: str

def should_use_tools(state: ConditionalToolState) -> str:
    """Decide whether tools are needed."""
    query = state["query"].lower()
    
    needs_tools = any(
        word in query 
        for word in ["weather", "stock", "email", "current", "today"]
    )
    
    return "use_tools" if needs_tools else "direct_response"

def with_tools(state: ConditionalToolState) -> dict:
    """Process with tool calling."""
    # Call model with tools bound
    response = model_with_tools.invoke(state["query"])
    return {"result": response.content, "use_tools": True}

def without_tools(state: ConditionalToolState) -> dict:
    """Process without tools."""
    response = model.invoke(state["query"])
    return {"result": response.content, "use_tools": False}

builder = StateGraph(ConditionalToolState)
builder.add_node("route", should_use_tools)
builder.add_node("with_tools", with_tools)
builder.add_node("without_tools", without_tools)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    should_use_tools,
    {
        "use_tools": "with_tools",
        "direct_response": "without_tools"
    }
)
builder.add_edge("with_tools", END)
builder.add_edge("without_tools", END)

conditional_tool_graph = builder.compile()

# Test
result = conditional_tool_graph.invoke({"query": "What's the weather?"})
print("Used tools:", result["use_tools"])  # True

result = conditional_tool_graph.invoke({"query": "Tell me a joke"})
print("Used tools:", result["use_tools"])  # False
```

