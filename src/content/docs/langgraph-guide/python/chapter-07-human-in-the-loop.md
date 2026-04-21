---
title: "Chapter 7 — Human-in-the-Loop"
description: "Pause execution for approval, build multi-stage review workflows, and iterate interactively with a human reviewer."
framework: langgraph
language: python
sidebar:
  label: "7 · Human-in-the-loop"
  order: 7
---

# Chapter 7 — Human-in-the-Loop

**What you'll learn:** how to pause your graph mid-execution to collect input, approval, or corrections from a human — the `interrupt()` and `Command(resume=...)` primitives, multi-stage approval workflows, and interactive debugging.

**Time:** ~25 minutes.

> Prereqs: [Chapter 5 — Memory & persistence](/langgraph-guide/python/chapter-05-memory/). Interrupts require a checkpointer.

> **Note on code snippets.** The examples below are **illustrative fragments** focused on the interrupt/resume pattern itself. They assume the standard imports from earlier chapters (`StateGraph`, `START`, `END`, `InMemorySaver`, `TypedDict`, `Annotated`) and elide helper stubs (`stream_events`, `refine_node`, `finalize_node`, `process`) — substitute your own. For end-to-end runnable HITL agents see the [Recipes collection](/langgraph-guide/python/langgraph_recipes/).

## Human-in-the-Loop

### Basic Interrupts

Pause execution and request human input:


```python
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    action: str
    amount: float
    approved: bool
    approval_reason: str

def request_approval(state: ApprovalState) -> dict:
    """Pause and ask human for approval."""
    
    # Interrupt with information for human
    result = interrupt({
        "action": state["action"],
        "amount": state["amount"],
        "message": f"Approve {state['action']} for ${state['amount']}?"
    })
    
    # result contains human's response
    return {
        "approved": result.get("approved", False),
        "approval_reason": result.get("reason", "")
    }

def execute_action(state: ApprovalState) -> dict:
    """Execute if approved."""
    if state["approved"]:
        return {"action": f"Executed {state['action']}"}
    else:
        return {"action": f"Rejected {state['action']}"}

# Build with interrupts
builder = StateGraph(ApprovalState)
builder.add_node("request_approval", request_approval)
builder.add_node("execute", execute_action)

builder.add_edge(START, "request_approval")
builder.add_edge("request_approval", "execute")
builder.add_edge("execute", END)

# MUST compile with checkpointer for interrupts
checkpointer = InMemorySaver()
approval_graph = builder.compile(checkpointer=checkpointer)

# Usage
config = {"configurable": {"thread_id": "approval-1"}}

# Start - will interrupt
events = []
for event in approval_graph.stream(
    {"action": "transfer", "amount": 500.00},
    config=config
):
    events.append(event)
    
print(events)
# Output: [{'__interrupt__': (Interrupt(...), )}]

# Check if interrupted
state = approval_graph.get_state(config)
if state.next == ("__interrupt__",):
    print("Waiting for human approval")
    
    # Human decides
    human_decision = {
        "approved": True,
        "reason": "Amount looks reasonable"
    }
    
    # Resume with decision
    resume_events = list(approval_graph.stream(
        Command(resume=human_decision),
        config=config
    ))
    
    print(resume_events)  # Graph continues
```


### Multi-Step Approval Workflow


```python
from enum import Enum

class ApprovalStage(Enum):
    INITIAL_REVIEW = "initial"
    COMPLIANCE_CHECK = "compliance"
    FINAL_APPROVAL = "final"

class WorkflowApprovalState(TypedDict):
    action: str
    amount: float
    approval_stage: ApprovalStage
    approvals: Annotated[dict, lambda x, y: {**x, **y}]

def initial_review_node(state: WorkflowApprovalState) -> dict:
    """First level approval."""
    
    approval = interrupt({
        "stage": "INITIAL",
        "question": f"Review {state['action']} for ${state['amount']}?",
        "reviewer_type": "manager"
    })
    
    return {
        "approvals": {"initial": approval.get("approved")},
        "approval_stage": ApprovalStage.COMPLIANCE_CHECK
    }

def compliance_check_node(state: WorkflowApprovalState) -> dict:
    """Second level - compliance."""
    
    # Only ask if initial approved
    if not state["approvals"].get("initial"):
        return {
            "approval_stage": ApprovalStage.FINAL_APPROVAL,
            "approvals": {"compliance": False}
        }
    
    approval = interrupt({
        "stage": "COMPLIANCE",
        "question": "Compliance clearance needed",
        "reviewer_type": "compliance_officer"
    })
    
    return {
        "approvals": {"compliance": approval.get("approved")},
        "approval_stage": ApprovalStage.FINAL_APPROVAL
    }

def final_approval_node(state: WorkflowApprovalState) -> dict:
    """Executive final approval."""
    
    all_approved = all(state["approvals"].values())
    
    if not all_approved:
        return {"approvals": {"final": False}}
    
    approval = interrupt({
        "stage": "FINAL",
        "question": "Executive approval required",
        "reviewer_type": "executive"
    })
    
    return {"approvals": {"final": approval.get("approved")}}

def execute_if_approved(state: WorkflowApprovalState) -> dict:
    """Only run if all approvals granted."""
    
    all_approved = all(state["approvals"].values())
    
    if all_approved:
        # Execute action
        return {"action": f"EXECUTED: {state['action']}"}
    else:
        return {"action": f"REJECTED: {state['action']}"}

# Build workflow
builder = StateGraph(WorkflowApprovalState)
builder.add_node("initial", initial_review_node)
builder.add_node("compliance", compliance_check_node)
builder.add_node("final", final_approval_node)
builder.add_node("execute", execute_if_approved)

builder.add_edge(START, "initial")
builder.add_edge("initial", "compliance")
builder.add_edge("compliance", "final")
builder.add_edge("final", "execute")
builder.add_edge("execute", END)

approval_workflow = builder.compile(checkpointer=InMemorySaver())

# Multi-stage execution
config = {"configurable": {"thread_id": "multi-approval-1"}}

# Stage 1
stream_events(approval_workflow.stream(
    {"action": "hire", "amount": 80000},
    config=config
))

# Resume with manager approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))

# Resume with compliance approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))

# Resume with executive approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))
```


### Interactive Debugging


```python
class DebugState(TypedDict):
    data: str
    step_result: str
    needs_adjustment: bool

def step_node(state: DebugState) -> dict:
    """Process data."""
    
    result = process(state["data"])
    
    # Ask if result is acceptable
    feedback = interrupt({
        "step": "Process",
        "result": result,
        "question": "Is this result acceptable? (yes/no/modify)"
    })
    
    if feedback["action"] == "modify":
        result = feedback["modified_result"]
        needs_adjustment = True
    else:
        needs_adjustment = feedback["action"] != "yes"
    
    return {
        "step_result": result,
        "needs_adjustment": needs_adjustment
    }

def decide_continue(state: DebugState) -> str:
    """Route based on feedback."""
    return "refine" if state["needs_adjustment"] else "finalize"

# Build interactive debug workflow
builder = StateGraph(DebugState)
builder.add_node("process", step_node)
builder.add_node("refine", refine_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    decide_continue,
    {"refine": "refine", "finalize": "finalize"}
)
builder.add_edge("refine", "process")
builder.add_edge("finalize", END)

debug_workflow = builder.compile(checkpointer=InMemorySaver())

# Interactive use
config = {"configurable": {"thread_id": "debug-session"}}

# Step through with feedback
stream_events(debug_workflow.stream(
    {"data": "raw_input"},
    config=config
))

# Human reviews and responds with modifications
stream_events(debug_workflow.stream(
    Command(resume={"action": "modify", "modified_result": "adjusted_output"}),
    config=config
))
```


