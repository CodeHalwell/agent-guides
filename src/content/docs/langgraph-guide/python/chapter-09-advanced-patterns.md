---
title: "Chapter 9 — Advanced Patterns"
description: "ReAct, Tree-of-Thoughts, self-reflection, structured output, node caching, deferred nodes, Command tool, templates, and the Functional API."
framework: langgraph
language: python
sidebar:
  label: "9 · Advanced patterns"
  order: 9
---

# Chapter 9 — Advanced Patterns

**What you'll learn:** the patterns you reach for when simple graphs aren't enough. Reasoning-action loops (ReAct), multi-path exploration (Tree-of-Thoughts), self-critique (Reflection), validation loops (Structured output), plus v1 features — node caching, deferred nodes, the Command tool for edgeless flows, templates, and the declarative Functional API.

**Time:** ~40 minutes. Most of this is reference — skim for patterns you need.

> Prereqs: [Chapter 3 — Multi-agent systems](/langgraph-guide/python/chapter-03-multi-agent/) and [Chapter 4 — Tools](/langgraph-guide/python/chapter-04-tools/).

## Advanced Patterns

### Pattern 1: ReAct (Reasoning + Acting)

The Reflection-Action pattern for autonomous agents, now built with modern LangChain components.

```python
# Note: AgentExecutor and create_tool_calling_agent require `pip install langchain langchain-anthropic`
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

@tool
def calculator(expression: str) -> str:
    """Calculate expression."""
    return str(eval(expression))

tools = [search_web, calculator]

# Create the ReAct agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful research assistant. Think before acting."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
react_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Use it - the AgentExecutor automatically handles the ReAct loop
result = react_agent.invoke({
    "input": "Research population of Tokyo and calculate 15% of that",
    "chat_history": []
})

print(result["output"])
```

### Pattern 2: Tree-of-Thoughts

Explore multiple reasoning paths:


```python
from langgraph.types import Send

class ThoughtState(TypedDict):
    question: str
    thoughts: Annotated[list[dict], lambda x, y: x + y]
    best_thought: dict
    final_answer: str

def generate_thoughts(state: ThoughtState) -> list[Send]:
    """Generate multiple solution approaches."""
    
    num_paths = 3
    returns = []
    
    for i in range(num_paths):
        returns.append(
            Send("explore_thought", {
                "question": state["question"],
                "path_number": i
            })
        )
    
    return returns

def explore_thought(state: ThoughtState) -> dict:
    """Explore one reasoning path."""
    
    prompt = f"""
    Question: {state['question']}
    Path #{state.get('path_number', 0)}
    
    Provide your reasoning for this specific approach.
    """
    
    response = model.invoke(prompt)
    
    return {
        "thoughts": [{
            "path": state.get("path_number"),
            "reasoning": response.content,
            "quality_score": 0.8  # Could be evaluated
        }]
    }

def select_best(state: ThoughtState) -> dict:
    """Select the best thought."""
    
    if not state["thoughts"]:
        return {"best_thought": {}}
    
    best = max(state["thoughts"], key=lambda x: x.get("quality_score", 0))
    
    return {"best_thought": best}

def synthesize(state: ThoughtState) -> dict:
    """Synthesize best thought into answer."""
    
    best_reasoning = state["best_thought"].get("reasoning", "")
    
    prompt = f"""
    Best reasoning: {best_reasoning}
    
    Provide a final answer based on this reasoning.
    """
    
    response = model.invoke(prompt)
    
    return {"final_answer": response.content}

# Build tree-of-thoughts
builder = StateGraph(ThoughtState)
builder.add_node("generate", generate_thoughts)
builder.add_node("explore", explore_thought)
builder.add_node("select", select_best)
builder.add_node("synthesize", synthesize)

builder.add_conditional_edges(
    START,
    lambda _: "generate"
)
builder.add_conditional_edges(
    "generate",
    lambda _: ["explore"],
    ["explore"]
)
builder.add_edge("explore", "select")
builder.add_edge("select", "synthesize")
builder.add_edge("synthesize", END)

tot_graph = builder.compile()

# Use it
result = tot_graph.invoke({
    "question": "How should we approach climate change?"
})

print("Best thought:", result["best_thought"]["reasoning"])
print("Final answer:", result["final_answer"])
```


### Pattern 3: Self-Reflection

Agent critiques its own output:

```python
class ReflectionState(TypedDict):
    question: str
    initial_response: str
    critique: str
    refined_response: str
    reflection_count: int

def generate_response(state: ReflectionState) -> dict:
    """Generate initial response."""
    
    response = model.invoke(state["question"])
    
    return {
        "initial_response": response.content,
        "reflection_count": 0
    }

def self_critique(state: ReflectionState) -> dict:
    """Critique the response."""
    
    prompt = f"""
    Question: {state['question']}
    Response: {state['initial_response']}
    
    Critique this response. What could be improved?
    """
    
    critique = model.invoke(prompt)
    
    return {"critique": critique.content}

def should_refine(state: ReflectionState) -> str:
    """Decide if response needs refinement."""
    
    if state["reflection_count"] >= 2:
        return "done"
    
    # Check critique for issues
    if any(word in state["critique"].lower() 
           for word in ["incorrect", "missing", "unclear", "incomplete"]):
        return "refine"
    
    return "done"

def refine_response(state: ReflectionState) -> dict:
    """Create refined response based on critique."""
    
    prompt = f"""
    Original question: {state['question']}
    Your response: {state['initial_response']}
    Critique: {state['critique']}
    
    Provide an improved response addressing the critique.
    """
    
    refined = model.invoke(prompt)
    
    return {
        "refined_response": refined.content,
        "reflection_count": state["reflection_count"] + 1
    }

# Build reflection loop
builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_response)
builder.add_node("critique", self_critique)
builder.add_node("refine", refine_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")

builder.add_conditional_edges(
    "critique",
    should_refine,
    {"refine": "refine", "done": END}
)

builder.add_edge("refine", "critique")  # Loop back for re-critique

reflection_graph = builder.compile()

# Use it
result = reflection_graph.invoke({
    "question": "Explain quantum computing to a child"
})

print("Initial:", result["initial_response"])
print("Refined:", result.get("refined_response", "No refinement needed"))
print("Reflection iterations:", result["reflection_count"])
```

### Pattern 4: Structured Output with Validation

```python
from pydantic import BaseModel, field_validator

class ResearchOutput(BaseModel):
    """Structured research output."""
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence_score: float
    
    @field_validator('confidence_score')
    def score_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Must be between 0 and 1')
        return v

class StructuredState(TypedDict):
    topic: str
    raw_research: str
    structured_output: ResearchOutput
    validation_passed: bool
    errors: list[str]

def research_node(state: StructuredState) -> dict:
    """Conduct research."""
    
    result = model.invoke(f"Research: {state['topic']}")
    
    return {"raw_research": result.content}

def structure_output(state: StructuredState) -> dict:
    """Parse into structured format."""
    
    prompt = f"""
    Research content: {state['raw_research']}
    
    Extract into JSON with fields:
    - topic
    - key_findings (list)
    - sources (list)
    - confidence_score (0-1)
    """
    
    response = model.invoke(prompt)
    
    try:
        import json
        parsed = json.loads(response.content)
        output = ResearchOutput(**parsed)
        return {
            "structured_output": output,
            "validation_passed": True,
            "errors": []
        }
    except Exception as e:
        return {
            "validation_passed": False,
            "errors": [str(e)]
        }

def decide_next(state: StructuredState) -> str:
    """Route based on validation."""
    if state["validation_passed"]:
        return "success"
    else:
        return "retry"

def retry_node(state: StructuredState) -> dict:
    """Re-attempt with error context."""
    
    prompt = f"""
    Previous errors: {', '.join(state['errors'])}
    Retry research on: {state['topic']}
    """
    
    result = model.invoke(prompt)
    
    return {"raw_research": result.content}

# Build validation graph
builder = StateGraph(StructuredState)
builder.add_node("research", research_node)
builder.add_node("structure", structure_output)
builder.add_node("retry", retry_node)

builder.add_edge(START, "research")
builder.add_edge("research", "structure")

builder.add_conditional_edges(
    "structure",
    decide_next,
    {"success": END, "retry": "retry"}
)

builder.add_edge("retry", "structure")  # Loop back

validation_graph = builder.compile()

# Use it
result = validation_graph.invoke({
    "topic": "AI safety"
})

if result["validation_passed"]:
    output = result["structured_output"]
    print(f"Topic: {output.topic}")
    print(f"Confidence: {output.confidence_score}")
    print(f"Findings: {output.key_findings}")
```

### Pattern 5: Caching and Memoization


```python
from functools import lru_cache
from langgraph.store.memory import InMemoryStore

class CacheState(TypedDict):
    query: str
    result: str
    cache_hit: bool

# Simple LRU cache for expensive operations
@lru_cache(maxsize=128)
def expensive_operation(query: str) -> str:
    """Simulate expensive operation."""
    import time
    time.sleep(1)
    return f"Result for {query}"

async def cached_operation_node(
    state: CacheState,
    store: Annotated[InMemoryStore, InjectedStore]
) -> dict:
    """Check cache before executing."""
    
    query = state["query"]
    namespace = ("cache", "results")
    
    # Check cache
    cached = await store.aget(namespace, query)
    
    if cached:
        return {
            "result": cached.value,
            "cache_hit": True
        }
    
    # Execute and cache
    result = expensive_operation(query)
    
    await store.aput(
        namespace,
        query,
        {"result": result, "timestamp": datetime.now().isoformat()}
    )
    
    return {
        "result": result,
        "cache_hit": False
    }

# Build with caching
builder = StateGraph(CacheState)
builder.add_node("process", cached_operation_node)

caching_graph = builder.compile(store=InMemoryStore())

# Usage
config = {"configurable": {"thread_id": "cache-test"}}

# First call - hits expensive operation
result = caching_graph.invoke({"query": "expensive"}, config=config)
print("Cache hit:", result["cache_hit"])  # False

# Second call - uses cache
result = caching_graph.invoke({"query": "expensive"}, config=config)
print("Cache hit:", result["cache_hit"])  # True
```



---

## Node Caching (v1.0.3+)


#### Basic Node Caching

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from functools import lru_cache
import hashlib
import json

class State(TypedDict):
    query: str
    result: str
    cache_hit: bool

# Built-in cache decorator for nodes
from langgraph.cache import cache_node

@cache_node(ttl=3600, cache_key="query")  # Cache for 1 hour
def expensive_llm_call(state: State) -> dict:
    """Expensive LLM operation - cache results."""

    query = state["query"]

    # This will only execute if not in cache
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    response = model.invoke(query)

    return {
        "result": response.content,
        "cache_hit": False  # First time
    }

# With manual cache control
def query_with_cache(state: State) -> dict:
    """Manual cache implementation."""

    query = state["query"]

    # Generate cache key
    cache_key = hashlib.md5(query.encode()).hexdigest()

    # Check graph-level cache
    cached_result = graph.cache.get(cache_key)

    if cached_result:
        return {
            "result": cached_result,
            "cache_hit": True
        }

    # Execute expensive operation
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    response = model.invoke(query)

    # Store in cache
    graph.cache.set(cache_key, response.content, ttl=3600)

    return {
        "result": response.content,
        "cache_hit": False
    }

# Build graph with caching
builder = StateGraph(State)
builder.add_node("query", expensive_llm_call)
builder.add_edge(START, "query")
builder.add_edge("query", END)

# Compile with cache enabled
graph = builder.compile(
    checkpointer=InMemorySaver(),
    cache=True  # Enable built-in caching
)

# Usage
result1 = graph.invoke({"query": "What is LangGraph?"})
print(f"Cache hit: {result1['cache_hit']}")  # False

result2 = graph.invoke({"query": "What is LangGraph?"})
print(f"Cache hit: {result2['cache_hit']}")  # True (cached!)
```

#### Semantic Caching with Embeddings

```python
from langgraph.cache import SemanticCache
from langchain_openai import OpenAIEmbeddings

# Semantic cache - finds similar queries
semantic_cache = SemanticCache(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.95,  # 95% similarity required
    ttl=7200  # 2 hour TTL
)

@cache_node(cache=semantic_cache, cache_key="query")
def semantic_cached_query(state: State) -> dict:
    """Cache semantically similar queries."""

    query = state["query"]

    # Even if query is slightly different, cached result returned
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    response = model.invoke(query)

    return {"result": response.content}

# Different wordings hit the same cache
result1 = graph.invoke({"query": "What is LangGraph used for?"})
result2 = graph.invoke({"query": "What are LangGraph's use cases?"})  # Cache hit!
```

#### Cache Invalidation Strategies

```python
from langgraph.cache import CachePolicy

# Time-based invalidation
time_policy = CachePolicy(ttl=3600, refresh_on_access=True)

# Conditional invalidation
def should_invalidate(state: State, cached_result: dict) -> bool:
    """Custom invalidation logic."""

    # Invalidate if data is stale
    if cached_result.get("timestamp"):
        age = datetime.now() - cached_result["timestamp"]
        return age > timedelta(hours=1)

    return False

conditional_policy = CachePolicy(
    ttl=7200,
    invalidate_fn=should_invalidate
)

@cache_node(cache_policy=conditional_policy)
def cached_with_validation(state: State) -> dict:
    """Cache with custom invalidation."""

    return {
        "result": compute_result(state),
        "timestamp": datetime.now()
    }
```


---

## Deferred Nodes (v1.0.3+)

**Delay node execution** until all upstream paths complete. Essential for fan-in patterns where you need to wait for parallel tasks.

#### Basic Deferred Node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated

class WorkflowState(TypedDict):
    tasks: list[dict]
    results: Annotated[list[dict], lambda x, y: x + y]
    summary: str

def split_tasks(state: WorkflowState) -> list[Send]:
    """Fan-out: Create parallel tasks."""

    return [
        Send("worker", {"task_id": task["id"], "data": task["data"]})
        for task in state["tasks"]
    ]

def worker_node(state: WorkflowState) -> dict:
    """Process one task (runs in parallel)."""

    task_id = state.get("task_id")
    data = state.get("data")

    # Do work
    result = process_task(data)

    return {"results": [{"id": task_id, "result": result}]}

# DEFERRED NODE - waits for all workers
from langgraph.graph import deferred

@deferred(wait_for=["worker"])  # Wait for all worker executions
def aggregate_results(state: WorkflowState) -> dict:
    """Fan-in: Only runs after ALL workers complete."""

    # At this point, ALL worker results are available
    all_results = state["results"]

    # Aggregate
    summary = f"Processed {len(all_results)} tasks successfully"

    return {"summary": summary}

# Build graph
builder = StateGraph(WorkflowState)
builder.add_node("split", split_tasks)
builder.add_node("worker", worker_node)
builder.add_node("aggregate", aggregate_results)  # Deferred

builder.add_conditional_edges(START, lambda _: "split")
builder.add_conditional_edges("split", lambda _: ["worker"], ["worker"])
builder.add_edge("worker", "aggregate")  # Aggregate deferred until all workers done
builder.add_edge("aggregate", END)

graph = builder.compile()

# Execute
result = graph.invoke({
    "tasks": [
        {"id": 1, "data": "task-a"},
        {"id": 2, "data": "task-b"},
        {"id": 3, "data": "task-c"}
    ]
})

print(result["summary"])  # "Processed 3 tasks successfully"
```

#### Complex Deferred Pattern

```python
class ComplexState(TypedDict):
    main_task: str
    parallel_results: Annotated[list, lambda x, y: x + y]
    validation_results: Annotated[list, lambda x, y: x + y]
    final_output: str

def parallel_processor(state: ComplexState) -> dict:
    """Multiple parallel processors."""
    return {"parallel_results": [process(state["main_task"])]}

def validator(state: ComplexState) -> dict:
    """Multiple parallel validators."""
    return {"validation_results": [validate(state["main_task"])]}

@deferred(wait_for=["parallel_processor", "validator"])
def final_decision(state: ComplexState) -> dict:
    """Waits for BOTH parallel processors AND validators."""

    all_processing_done = state["parallel_results"]
    all_validations_done = state["validation_results"]

    # Make final decision with complete information
    if all(v["valid"] for v in all_validations_done):
        return {"final_output": "All validations passed"}
    else:
        return {"final_output": "Validation failed"}

# The final_decision node will not execute until:
# 1. ALL parallel_processor instances complete
# 2. ALL validator instances complete
```

---

## Tools with State Updates (v1.0.3+)

**Tools can directly update graph state** - no need to return values and merge them. Enables more powerful and dynamic tool behaviors.

#### Tools with State Updates

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.types import StateUpdate

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: dict
    cart_items: list[dict]
    total_price: float

# Traditional tool - returns value
@tool
def old_add_to_cart(product_id: str, quantity: int) -> str:
    """Old way: Return string that gets added to messages."""
    return f"Added {quantity} of {product_id} to cart"

# NEW: Tool with state updates
@tool(updates_state=True)
def add_to_cart(
    product_id: str,
    quantity: int,
    state: AgentState
) -> StateUpdate:
    """New way: Directly update graph state."""

    # Access current state
    current_cart = state.get("cart_items", [])
    current_total = state.get("total_price", 0.0)

    # Fetch product info
    product = get_product(product_id)
    price = product["price"] * quantity

    # Update cart
    current_cart.append({
        "product_id": product_id,
        "name": product["name"],
        "quantity": quantity,
        "price": price
    })

    # Return state updates directly
    return StateUpdate(
        cart_items=current_cart,
        total_price=current_total + price,
        messages=[{
            "role": "tool",
            "content": f"Added {quantity}x {product['name']} (${price}) to cart. Total: ${current_total + price}"
        }]
    )

@tool(updates_state=True)
def update_user_preferences(
    preference_key: str,
    preference_value: Any,
    state: AgentState
) -> StateUpdate:
    """Tool that modifies user context."""

    context = state.get("user_context", {})
    context["preferences"] = context.get("preferences", {})
    context["preferences"][preference_key] = preference_value

    return StateUpdate(
        user_context=context,
        messages=[{
            "role": "tool",
            "content": f"Updated preference: {preference_key} = {preference_value}"
        }]
    )

@tool(updates_state=True)
def search_and_filter(
    query: str,
    filters: dict,
    state: AgentState
) -> StateUpdate:
    """Tool that performs search and updates state with results."""

    # Perform search
    results = search_api(query, filters)

    # Update state with search metadata
    return StateUpdate(
        search_results=results,
        last_search_query=query,
        search_count=state.get("search_count", 0) + 1,
        messages=[{
            "role": "tool",
            "content": f"Found {len(results)} results for '{query}'"
        }]
    )

# Use tools with state updates
tools = [add_to_cart, update_user_preferences, search_and_filter]

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """Agent that can call tools."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode automatically handles state updates
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

graph = builder.compile()

# Tool calls directly modify cart_items and total_price!
result = graph.invoke({
    "messages": [{"role": "user", "content": "Add 2 laptops to my cart"}]
})

print(f"Cart: {result['cart_items']}")
print(f"Total: ${result['total_price']}")
```

#### Advanced State Update Patterns

```python
@tool(updates_state=True)
def multi_step_tool(query: str, state: AgentState) -> StateUpdate:
    """Tool that performs multiple state updates."""

    # Step 1: Search
    results = search(query)

    # Step 2: Filter based on user preferences
    user_prefs = state.get("user_context", {}).get("preferences", {})
    filtered = [r for r in results if matches_preferences(r, user_prefs)]

    # Step 3: Sort by relevance
    sorted_results = sort_by_relevance(filtered, query)

    # Update multiple state fields
    return StateUpdate(
        search_results=sorted_results,
        last_query=query,
        result_count=len(sorted_results),
        search_history=state.get("search_history", []) + [query],
        messages=[{
            "role": "tool",
            "content": f"Found {len(sorted_results)} relevant results"
        }]
    )

@tool(updates_state=True, async_update=True)
async def async_state_update(url: str, state: AgentState) -> StateUpdate:
    """Async tool with state updates."""

    # Async operations
    data = await fetch_url(url)
    processed = await process_data(data)

    return StateUpdate(
        fetched_data=processed,
        last_fetch_url=url,
        fetch_timestamp=datetime.now().isoformat()
    )
```

---

## Command Tool for Edgeless Flows (v1.0.3+)

**Build dynamic, edgeless agent flows** where the agent decides its own path. No predefined edges - the agent uses a special "command" tool to control execution flow.

#### Basic Command Tool Usage

```python
from langgraph.prebuilt import create_react_agent
# Note: command_tool is a conceptual pattern; route control via state fields and conditional edges
from langchain_core.tools import tool

# Regular tools
@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    return str(eval(expression))

# Command tool - agent controls flow
@command_tool
def route_to_specialist(specialist: str, task: str) -> dict:
    """
    Route task to a specialist agent.

    Args:
        specialist: One of ["researcher", "analyst", "writer"]
        task: Task description for the specialist
    """
    return {
        "next_agent": specialist,
        "task": task,
        "routed_at": datetime.now().isoformat()
    }

@command_tool
def finish_conversation(summary: str) -> dict:
    """
    End the conversation with a summary.

    Args:
        summary: Final summary of what was accomplished
    """
    return {
        "finished": True,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }

# Create agent with command tools
tools = [search_web, calculate, route_to_specialist, finish_conversation]

agent = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=tools,
    command_tools=[route_to_specialist, finish_conversation]
)

# Agent dynamically controls its own flow
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Research AI trends, analyze the data, and write a report"
    }]
})

# Agent will:
# 1. Use route_to_specialist("researcher", "Research AI trends")
# 2. Use route_to_specialist("analyst", "Analyze the research data")
# 3. Use route_to_specialist("writer", "Write report from analysis")
# 4. Use finish_conversation("Report completed")
```

#### Advanced Command Tool Patterns

```python
from langgraph.graph import StateGraph, START, END
# CommandRouter pattern: use conditional edges with a routing function instead

class DynamicState(TypedDict):
    messages: Annotated[list, add_messages]
    current_phase: str
    data: dict
    commands_executed: list[dict]

@command_tool
def start_research_phase(topic: str) -> dict:
    """Begin research phase on a topic."""
    return {
        "command": "set_phase",
        "phase": "research",
        "topic": topic
    }

@command_tool
def start_analysis_phase(research_data: dict) -> dict:
    """Begin analysis phase with research data."""
    return {
        "command": "set_phase",
        "phase": "analysis",
        "data": research_data
    }

@command_tool
def start_synthesis_phase(analysis_results: dict) -> dict:
    """Begin synthesis phase with analysis."""
    return {
        "command": "set_phase",
        "phase": "synthesis",
        "data": analysis_results
    }

@command_tool
def loop_back(reason: str) -> dict:
    """Loop back to previous phase if needed."""
    return {
        "command": "loop_back",
        "reason": reason
    }

# Command router handles command tool outputs
command_router = CommandRouter(
    commands={
        "set_phase": lambda cmd: cmd["phase"],
        "loop_back": lambda cmd: "analysis"  # Go back
    },
    default="research"
)

def agent_node(state: DynamicState) -> dict:
    """Agent that uses command tools to control flow."""

    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    # Bind all tools including command tools
    model_with_tools = model.bind_tools([
        search_web,
        start_research_phase,
        start_analysis_phase,
        start_synthesis_phase,
        loop_back
    ])

    response = model_with_tools.invoke(state["messages"])

    # Check for command tool calls
    if hasattr(response, 'tool_calls'):
        for tool_call in response.tool_calls:
            if tool_call["name"].startswith("start_") or tool_call["name"] == "loop_back":
                # This is a command tool - extract command
                result = execute_tool(tool_call)
                return {
                    "messages": [response],
                    "current_phase": result.get("phase", state["current_phase"]),
                    "commands_executed": state.get("commands_executed", []) + [result]
                }

    return {"messages": [response]}

# No predefined edges - agent controls flow with command tools
builder = StateGraph(DynamicState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")

# Router uses command tool outputs to decide next node
def route_based_on_command(state: DynamicState) -> str:
    """Route based on last command executed."""

    if not state.get("commands_executed"):
        return "agent"

    last_command = state["commands_executed"][-1]

    if last_command.get("command") == "set_phase":
        return last_command["phase"]
    elif last_command.get("command") == "loop_back":
        return "analysis"
    else:
        return END

builder.add_conditional_edges(
    "agent",
    route_based_on_command,
    {
        "research": "agent",
        "analysis": "agent",
        "synthesis": "agent",
        END: END
    }
)

graph = builder.compile()

# Agent dynamically controls its workflow
result = graph.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze the impact of AI on healthcare"
    }]
})
```

---

## LangGraph Templates

**Pre-built graph templates** for common agentic patterns. Start building instantly with production-ready templates.

#### Available Templates

```bash
# List available templates
langgraph template list

# Templates:
# - react-agent: ReAct agent with tool calling
# - chatbot: Simple conversational agent
# - research-agent: Multi-source research agent
# - rag-agent: RAG system with quality control
# - supervisor: Multi-agent supervisor pattern
# - parallel-workers: Fan-out/fan-in pattern
# - human-in-loop: Approval workflow
# - self-reflection: Self-critiquing agent
```

#### Using Templates

```bash
# Create new project from template
langgraph template create my-agent --template react-agent

cd my-agent

# Project structure created:
# my-agent/
#   ├── agent.py          # Main agent implementation
#   ├── state.py          # State schemas
#   ├── tools.py          # Tool definitions
#   ├── config.py         # Configuration
#   ├── langgraph.json    # LangGraph CLI config
#   ├── requirements.txt
#   └── README.md

# Customize the generated code
# Then run:
langgraph run
```

#### Template Customization

```python
# agent.py (generated from template)
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from tools import get_tools

# Template provides structure, you customize
def create_agent():
    """Create ReAct agent (generated from template)."""

    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0  # Customize
    )

    tools = get_tools()  # Customize tools

    # Template handles graph structure
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a helpful assistant."  # Customize prompt
    )

    return agent

# Run agent
graph = create_agent()
```

#### Creating Custom Templates

```python
# Create template directory structure
langgraph template init my-custom-template

# Define template structure
# my-custom-template/
#   ├── template.yaml      # Template metadata
#   ├── {{cookiecutter.project_name}}/
#   │   ├── agent.py
#   │   ├── state.py
#   │   └── ...

# template.yaml
# name: my-custom-template
# description: Custom agent template
# variables:
#   project_name:
#     type: string
#     default: my-agent
#   model:
#     type: choice
#     choices: [claude-3-5-sonnet, gpt-4]

# Share template
langgraph template publish my-custom-template

---

## Functional API (LangGraph 1.0)

A simpler Python-native way to build workflows with automatic parallelization:


```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional

# Define parallelizable tasks
@task
def fetch_user_data(user_id: str) -> dict:
    """Get user info."""
    return {"user_id": user_id, "name": "Alice"}

@task
def fetch_orders(user_id: str) -> list[dict]:
    """Get user orders."""
    return [{"id": "1", "total": 99.99}]

@task
async def generate_recommendations(user_data: dict, orders: list) -> list[str]:
    """Generate recommendations (can be async)."""
    return ["Product A", "Product B"]

# Define entrypoint with automatic parallelization
@entrypoint(checkpointer=InMemorySaver())
def build_dashboard(user_id: str, *, previous: Optional[dict] = None) -> dict:
    """
    Build dashboard with parallel data fetching.
    
    Args:
        user_id: User to fetch data for
        previous: Return value from last invocation (enables state)
    
    Returns:
        Complete dashboard data
    """
    
    # Launch tasks in parallel - immediately get futures
    user_future = fetch_user_data(user_id)
    orders_future = fetch_orders(user_id)
    
    # Block and wait for results
    user_data = user_future.result()
    orders = orders_future.result()
    
    # Now generate recommendations using results
    recs_future = generate_recommendations(user_data, orders)
    recommendations = recs_future.result()
    
    # Can interrupt for human approval
    approved = interrupt({
        "recommendations": recommendations,
        "question": "Approve these recommendations?"
    })
    
    return {
        "user": user_data,
        "orders": orders,
        "recommendations": recommendations if approved else [],
        "status": "approved" if approved else "rejected"
    }

# Execute
config = {"configurable": {"thread_id": "user-session-1"}}

# Initial run - interrupts for approval
for result in build_dashboard.stream("user-123", config):
    print(result)

# Resume after human approval
for result in build_dashboard.stream(Command(resume=True), config):
    print(result)

# With previous state for stateful workflows
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Optional[int] = None) -> str:
    """Accumulate counter."""
    current = (previous or 0) + increment
    return f"Counter: {current}"

config = {"configurable": {"thread_id": "counter"}}
counter.invoke(5, config)    # "Counter: 5"
counter.invoke(3, config)    # "Counter: 8" (5+3)
```
