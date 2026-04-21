---
title: "Chapter 5 — Memory & Persistence"
description: "Short-term state via checkpointers, long-term memory via Store, and cross-thread memory that survives across sessions."
framework: langgraph
language: python
sidebar:
  label: "5 · Memory & persistence"
  order: 5
---

# Chapter 5 — Memory & Persistence

**What you'll learn:** LangGraph's two memory tiers. **Checkpointers** save graph state at each step so you can resume after a failure. **Stores** provide durable key/value storage (and optional semantic search) that survives across threads and users. You'll also see the **cross-thread memory** pattern that lets one conversation learn from another.

**Time:** ~25 minutes.

> Prereqs: [Chapter 1 — Setup & core concepts](/langgraph-guide/python/chapter-01-setup-and-core-concepts/).

## Memory & Persistence

### Short-Term Memory: Checkpointers

Checkpointers save graph state automatically at each step. Resume from failures.

#### In-Memory (Development)

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# State persists within this Python process only
# Useful for development & testing
```

#### SQLite (Local Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Or in-memory SQLite
checkpointer = SqliteSaver.from_conn_string(":memory:")

graph = builder.compile(checkpointer=checkpointer)
```

#### PostgreSQL (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

# Async version
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async_checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

graph = builder.compile(checkpointer=checkpointer)
```

### Using Checkpoints


```python
config = {"configurable": {"thread_id": "user-123"}}

# First invocation
result = graph.invoke(
    {"query": "Start process"},
    config=config
)

# Check current state
current_state = graph.get_state(config)
print(f"Next node: {current_state.next}")
print(f"Values: {current_state.values}")
print(f"Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")

# Continue in same thread - state restored from checkpoint
result = graph.invoke(
    {"query": "Continue"},
    config=config
)

# Get state history (time-travel debugging)
history = graph.get_state_history(config)

for i, checkpoint in enumerate(history):
    cp_id = checkpoint.config['configurable']['checkpoint_id']
    print(f"Step {i}: {cp_id}")
    print(f"  State: {checkpoint.values}")

# Resume from specific checkpoint (time-travel)
old_checkpoint_id = history[1].config['configurable']['checkpoint_id']
time_travel_config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": old_checkpoint_id
    }
}

# Continue from that point in history
result = graph.invoke(
    {"query": "New direction"},
    config=time_travel_config
)
```


### Long-Term Memory: Store

Store provides cross-thread, persistent key-value storage with hierarchical namespaces:

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore

# In-memory for development
store = InMemoryStore()

# PostgreSQL for production (with vector search)
store = AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

# Store operations
namespace = ("users", "user-123", "preferences")

# Put data
await store.aput(
    namespace=namespace,
    key="theme",
    value={"dark_mode": True, "language": "en"}
)

# Get data
item = await store.aget(namespace, "theme")
print(item.value)  # {"dark_mode": True, ...}

# List all in namespace
items = await store.asearch(namespace_prefix=namespace)
for item in items:
    print(f"{item.key}: {item.value}")

# Delete
await store.adelete(namespace, "theme")

# Store with vector search for semantic retrieval
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store_with_search = AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db",
    embeddings=embeddings
)

# Store documents with embeddings
await store_with_search.aput(
    namespace=("docs", "kb"),
    key="api-guide",
    value={
        "title": "API Guide",
        "content": "LangGraph provides APIs for building stateful agents..."
    },
    index=["content"]  # Fields to embed
)

# Semantic search
results = await store_with_search.asearch(
    namespace_prefix=("docs",),
    query="how to build agents",
    limit=5
)

for result in results:
    print(f"Score: {result.score}, {result.value['title']}")
```

### Injecting Store into Nodes

Use LangGraph's dependency injection:

```python
from langgraph.prebuilt import InjectedStore
from typing import Annotated

def personalization_node(
    state: State,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Node that accesses store automatically."""
    user_id = state["user_id"]
    
    # Retrieve preferences
    namespace = ("users", user_id, "prefs")
    prefs_item = await store.aget(namespace, "theme")
    prefs = prefs_item.value if prefs_item else {}
    
    # Update if interaction changes preferences
    if state.get("user_voted_dark"):
        await store.aput(
            namespace,
            "theme",
            {"dark_mode": True, "last_updated": datetime.now().isoformat()}
        )
    
    return {"user_preferences": prefs}

# Compile with store
builder = StateGraph(State)
builder.add_node("personalize", personalization_node)

graph = builder.compile(store=store)
```

### Complete Memory Example


```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from datetime import datetime

class MemoryState(TypedDict):
    user_id: str
    message: str
    response: str
    conversation_history: Annotated[list, add_messages]

def store_memory_node(
    state: MemoryState,
    store: Annotated[InMemoryStore, InjectedStore]
) -> dict:
    """Store user preferences and conversation summary."""
    
    # Extract user preferences from conversation
    namespace = ("users", state["user_id"], "memory")
    
    # Save conversation turn
    await store.aput(
        namespace,
        f"turn-{datetime.now().isoformat()}",
        {
            "user_message": state["message"],
            "bot_response": state["response"]
        }
    )
    
    # Update user profile based on interactions
    profile_key = "profile"
    profile = await store.aget(namespace, profile_key)
    existing = profile.value if profile else {}
    
    updated_profile = {
        **existing,
        "total_turns": existing.get("total_turns", 0) + 1,
        "last_interaction": datetime.now().isoformat()
    }
    
    await store.aput(namespace, profile_key, updated_profile)
    
    return {}

# Build with memory
checkpointer = SqliteSaver.from_conn_string("memory.db")
store = InMemoryStore()

builder = StateGraph(MemoryState)
builder.add_node("respond", respond_node)
builder.add_node("remember", store_memory_node)

builder.add_edge(START, "respond")
builder.add_edge("respond", "remember")
builder.add_edge("remember", END)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)

# Use with persistence
config = {"configurable": {"thread_id": "user-alice"}}

for i in range(3):
    result = graph.invoke(
        {"user_id": "alice", "message": f"Message {i}"},
        config=config
    )
    print(result["response"])
    
# Multi-turn conversations remembered automatically
```



---

## Cross-Thread Memory (v1.0.3+)

**Share memory across conversation threads** for true multi-session context. Perfect for user preferences, learned information, and global knowledge.

#### Basic Cross-Thread Memory

```python
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.prebuilt import InjectedStore
from typing import Annotated

class ChatState(TypedDict):
    user_id: str
    message: str
    messages: Annotated[list, add_messages]
    user_profile: dict
    learned_facts: list[dict]

# Cross-thread store
store = AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

async def load_user_context(
    state: ChatState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Load user data that persists across ALL threads."""

    user_id = state["user_id"]

    # Namespace for cross-thread data
    user_namespace = ("global", "users", user_id)

    # Load persistent user profile
    profile_item = await store.aget(user_namespace, "profile")
    profile = profile_item.value if profile_item else {
        "name": "User",
        "preferences": {},
        "total_conversations": 0
    }

    # Load learned facts from ALL previous conversations
    facts_items = await store.asearch(
        namespace_prefix=(user_namespace + ("facts",)),
        limit=50
    )

    learned_facts = [item.value for item in facts_items]

    return {
        "user_profile": profile,
        "learned_facts": learned_facts
    }

async def chat_with_memory(
    state: ChatState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Generate response using cross-thread context."""

    # Build context from learned facts
    facts_context = "\n".join([
        f"- {fact['content']}"
        for fact in state["learned_facts"]
    ])

    system_prompt = f"""
    You are chatting with {state['user_profile']['name']}.

    What you know about this user from previous conversations:
    {facts_context}

    Use this context naturally in your responses.
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    response = model.invoke(messages)

    return {"messages": [response]}

async def save_learnings(
    state: ChatState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Extract and save new facts for future threads."""

    user_id = state["user_id"]
    user_namespace = ("global", "users", user_id)

    # Extract new information from conversation
    last_user_message = state["messages"][-2].content
    last_bot_message = state["messages"][-1].content

    # Use LLM to extract learnable facts
    
    extraction_prompt = f"""
    Extract any facts learned about the user from this exchange:
    User: {last_user_message}
    Assistant: {last_bot_message}

    Return facts as JSON array: [{{"content": "fact", "confidence": 0.9}}]
    """
    

    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    facts_response = model.invoke(extraction_prompt)

    import json
    try:
        new_facts = json.loads(facts_response.content)

        # Store each fact in cross-thread namespace
        for fact in new_facts:
            fact_id = f"fact-{uuid.uuid4().hex[:8]}"
            await store.aput(
                user_namespace + ("facts",),
                fact_id,
                {
                    "content": fact["content"],
                    "confidence": fact.get("confidence", 0.8),
                    "learned_at": datetime.now().isoformat(),
                    "thread_id": state.get("thread_id", "unknown")
                }
            )
    except:
        pass  # Failed to extract facts

    # Update conversation count
    profile = state["user_profile"]
    profile["total_conversations"] = profile.get("total_conversations", 0) + 1
    profile["last_seen"] = datetime.now().isoformat()

    await store.aput(user_namespace, "profile", profile)

    return {}

# Build graph with cross-thread memory
builder = StateGraph(ChatState)
builder.add_node("load_context", load_user_context)
builder.add_node("chat", chat_with_memory)
builder.add_node("save_learnings", save_learnings)

builder.add_edge(START, "load_context")
builder.add_edge("load_context", "chat")
builder.add_edge("chat", "save_learnings")
builder.add_edge("save_learnings", END)

# Compile with store for cross-thread memory
graph = builder.compile(
    checkpointer=InMemorySaver(),  # Thread-specific checkpoints
    store=store  # Cross-thread memory
)

# Thread 1
config1 = {"configurable": {"thread_id": "thread-1"}}
result1 = graph.invoke({
    "user_id": "alice",
    "message": "My favorite color is blue"
}, config=config1)

# Thread 2 - DIFFERENT conversation, but knows about blue!
config2 = {"configurable": {"thread_id": "thread-2"}}
result2 = graph.invoke({
    "user_id": "alice",
    "message": "What should I paint my room?"
}, config=config2)

# Response will reference "blue" learned from thread-1!
```

#### Team Memory - Shared Across Users

```python
async def team_knowledge_node(
    state: ChatState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Access team-wide knowledge base."""

    team_id = state.get("team_id")

    # Team-wide namespace (not user-specific)
    team_namespace = ("global", "teams", team_id, "knowledge")

    # Search team knowledge
    query = state["message"]
    relevant_docs = await store.asearch(
        namespace_prefix=team_namespace,
        query=query,
        limit=5
    )

    team_context = "\n".join([doc.value["content"] for doc in relevant_docs])

    return {"team_knowledge": team_context}

# Anyone on the team can access shared knowledge
```
