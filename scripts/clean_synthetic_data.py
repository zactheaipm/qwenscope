"""Clean synthetic training data: fix structural issues and remove train/eval leakage.

Issues addressed:
1. Consecutive assistant→assistant messages (merge into one)
2. Examples ending with tool role (drop — truncated)
3. Malformed tool_call objects (fix nesting structure)
4. Train/eval data leakage (deduplicate eval against train by user prompt)
5. Add missing tool_call_id / id fields for OpenAI format compliance
"""

from __future__ import annotations

import json
import hashlib
import uuid
import sys
from pathlib import Path


def _fix_tool_call(tc: dict) -> dict:
    """Normalize a tool_call to correct OpenAI nesting structure."""
    # Already correct: {"type": "function", "function": {"name": ..., "arguments": ...}}
    if "function" in tc and isinstance(tc["function"], dict) and "name" in tc["function"]:
        fn = tc["function"]
        # Ensure arguments is a dict, not None
        if fn.get("arguments") is None:
            fn["arguments"] = {}
        # Add id if missing
        if "id" not in tc:
            tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
        return tc

    # Flattened: {"type": "<tool_type>", "arguments": {...}}  or {"name": ..., "content": ...}
    name = tc.get("name", tc.get("type", "unknown"))
    if name in ("function",):
        name = "unknown"
    args = tc.get("arguments", tc.get("content", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args}
    if args is None:
        args = {}

    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        },
    }


def _merge_consecutive_assistants(messages: list[dict]) -> list[dict]:
    """Merge consecutive assistant messages into one.

    Pattern: assistant(text only) → assistant(tool_calls only)
    Result: single assistant with both content and tool_calls.
    """
    merged = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if (
            msg.get("role") == "assistant"
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "assistant"
        ):
            next_msg = messages[i + 1]
            # Merge: take content from first, tool_calls from second (or vice versa)
            content = msg.get("content") or next_msg.get("content")
            tool_calls = msg.get("tool_calls") or next_msg.get("tool_calls")
            merged_msg = {"role": "assistant"}
            if content:
                merged_msg["content"] = content
            else:
                merged_msg["content"] = None
            if tool_calls:
                merged_msg["tool_calls"] = tool_calls
            merged.append(merged_msg)
            i += 2
        else:
            merged.append(msg)
            i += 1
    return merged


def _add_tool_call_ids(messages: list[dict]) -> list[dict]:
    """Add id fields to tool_calls and tool_call_id to tool messages."""
    pending_ids: list[str] = []

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                tc = _fix_tool_call(tc)
                pending_ids.append(tc["id"])
                new_tcs.append(tc)
            msg["tool_calls"] = new_tcs
        elif msg.get("role") == "tool":
            if pending_ids:
                msg["tool_call_id"] = pending_ids.pop(0)
            else:
                msg["tool_call_id"] = f"call_{uuid.uuid4().hex[:12]}"
            # Ensure name field exists
            if "name" not in msg:
                msg["name"] = "unknown"

    return messages


def clean_examples(examples: list[dict]) -> list[dict]:
    """Clean a list of synthetic examples."""
    cleaned = []
    stats = {
        "total": len(examples),
        "dropped_truncated": 0,
        "merged_consecutive": 0,
        "fixed_tool_calls": 0,
    }

    for ex in examples:
        messages = ex.get("messages", [])

        # Drop examples that end with tool role (truncated)
        if messages and messages[-1].get("role") == "tool":
            stats["dropped_truncated"] += 1
            continue

        # Merge consecutive assistant messages
        orig_len = len(messages)
        messages = _merge_consecutive_assistants(messages)
        if len(messages) < orig_len:
            stats["merged_consecutive"] += 1

        # Add tool_call_id fields and fix malformed tool_calls
        messages = _add_tool_call_ids(messages)

        # Remove null tool_calls field from assistant messages that don't have tool calls
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls") is None:
                msg.pop("tool_calls", None)

        ex["messages"] = messages
        cleaned.append(ex)

    return cleaned, stats


def deduplicate_eval(
    train_examples: list[dict], eval_examples: list[dict]
) -> tuple[list[dict], int]:
    """Remove eval examples whose user prompt appears in training data."""
    train_prompts = set()
    for ex in train_examples:
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                # Use first 200 chars as key to catch near-duplicates too
                prompt = msg.get("content", "")[:200]
                train_prompts.add(hashlib.sha256(prompt.encode()).hexdigest())
                break

    deduped = []
    removed = 0
    for ex in eval_examples:
        user_prompt = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")[:200]
                break
        prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()
        if prompt_hash in train_prompts:
            removed += 1
        else:
            deduped.append(ex)

    return deduped, removed


def main():
    data_dir = Path("data/synthetic")

    # Load data
    print("Loading data...")
    with open(data_dir / "train_examples.jsonl") as f:
        train = [json.loads(line) for line in f]
    with open(data_dir / "eval_examples.jsonl") as f:
        eval_data = [json.loads(line) for line in f]

    print(f"  Train: {len(train)} examples")
    print(f"  Eval:  {len(eval_data)} examples")

    # Clean both splits
    print("\nCleaning train split...")
    train_clean, train_stats = clean_examples(train)
    print(f"  Dropped (truncated): {train_stats['dropped_truncated']}")
    print(f"  Merged consecutive:  {train_stats['merged_consecutive']}")

    print("\nCleaning eval split...")
    eval_clean, eval_stats = clean_examples(eval_data)
    print(f"  Dropped (truncated): {eval_stats['dropped_truncated']}")
    print(f"  Merged consecutive:  {eval_stats['merged_consecutive']}")

    # Deduplicate eval against train
    print("\nDeduplicating eval against train...")
    eval_deduped, n_removed = deduplicate_eval(train_clean, eval_clean)
    print(f"  Removed {n_removed} eval examples with prompts also in train")

    # Write cleaned files
    print("\nWriting cleaned data...")
    with open(data_dir / "train_examples.jsonl", "w") as f:
        for ex in train_clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(data_dir / "eval_examples.jsonl", "w") as f:
        for ex in eval_deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Update manifests
    for split, examples in [("train", train_clean), ("eval", eval_deduped)]:
        manifest = {
            "split": split,
            "n_total": len(examples),
            "n_per_key": len(examples) // 5,
            "n_keys": 5,
            "model": "deepseek-chat",
            "provider": "openai",
            "api_base_url": "https://api.deepseek.com/v1",
            "cleaned": True,
        }
        with open(data_dir / f"{split}_examples.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nFinal counts:")
    print(f"  Train: {len(train_clean)} (was {len(train)})")
    print(f"  Eval:  {len(eval_deduped)} (was {len(eval_data)})")
    print("\nDone.")


if __name__ == "__main__":
    main()
