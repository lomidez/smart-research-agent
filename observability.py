# observability.py
import json, os, time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langsmith import Client
from langsmith.run_trees import RunTree

# ---- Client & root run context ---------------------------------------------
_client = Client()  # uses LANGSMITH_* env vars
_root_ctx: ContextVar[Optional[RunTree]] = ContextVar("ls_root_ctx", default=None)

# Optional, configurable pricing (USD/token); override via env
# Example:
# export MODEL_PRICING_JSON='{"gpt-4o-mini":{"input":0.00015,"output":0.0006}}'
_MODEL_PRICING: Dict[str, Dict[str, float]] = {}
try:
    _MODEL_PRICING = json.loads(os.getenv("MODEL_PRICING_JSON", "{}")) or {}
except Exception:
    _MODEL_PRICING = {}

def _estimate_cost(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> float:
    if not model or model not in _MODEL_PRICING:
        return 0.0
    rates = _MODEL_PRICING[model]
    pin = float(rates.get("input", 0.0))
    pout = float(rates.get("output", 0.0))
    return (pin * float(prompt_tokens or 0)) + (pout * float(completion_tokens or 0))

# ---- Root run helpers -------------------------------------------------------
def ls_start_root_run(name: str, inputs: Dict[str, Any], tags: Optional[list[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> RunTree:
    """
    Start a LangSmith root run. Call this at the beginning of each user prompt handling.
    """
    root = RunTree(
        name=name,
        run_type="chain",  # top-level orchestration
        inputs=inputs or {},
        tags=tags or [],
        extra=metadata or {},
        project_name=os.getenv("LANGSMITH_PROJECT"),
    )
    _root_ctx.set(root)
    return root

def ls_end_root_run(outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    root = _root_ctx.get()
    if not root:
        return
    root.end(outputs=outputs or {}, error=error)
    root.post()  # send to LangSmith immediately

# ---- Child spans ------------------------------------------------------------
@contextmanager
def ls_span(name: str, run_type: str = "tool", inputs: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Create a child run (span). Use around external calls (web, arxiv, fetch, embeddings, etc.).
    """
    parent = _root_ctx.get()
    child = RunTree(
        name=name,
        run_type=run_type,    # "tool" | "llm" | "retriever" | "chain"
        inputs=inputs or {},
        extra=metadata or {},
        parent_run=parent,
        project_name=os.getenv("LANGSMITH_PROJECT"),
    )
    t0 = time.perf_counter()
    try:
        yield child
        dur_ms = int((time.perf_counter() - t0) * 1000)
        child.end(outputs={"duration_ms": dur_ms})
    except Exception as e:
        dur_ms = int((time.perf_counter() - t0) * 1000)
        child.end(error=str(e), outputs={"duration_ms": dur_ms})
        child.post()
        raise
    child.post()

def ls_log_llm_call(*, model: str, messages: Any, response_text: str, prompt_tokens: Optional[int], completion_tokens: Optional[int], total_tokens: Optional[int]):
    """
    Log an LLM child run with token & (optional) cost accounting.
    """
    parent = _root_ctx.get()
    child = RunTree(
        name=f"llm:{model}",
        run_type="llm",
        inputs={"messages": messages},
        parent_run=parent,
        project_name=os.getenv("LANGSMITH_PROJECT"),
    )
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    child.end(outputs={
        "response": response_text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 6),
        }
    })
    child.post()

    # Aggregate onto the root for a single “totals” view
    if parent:
        meta = parent.extra or {}
        totals = meta.get("totals") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
        totals["prompt_tokens"] += int(prompt_tokens or 0)
        totals["completion_tokens"] += int(completion_tokens or 0)
        totals["total_tokens"] += int(total_tokens or 0)
        totals["estimated_cost_usd"] = float(totals["estimated_cost_usd"]) + float(cost)
        meta["totals"] = totals
        parent.extra = meta  # keep in memory; posted when root ends
