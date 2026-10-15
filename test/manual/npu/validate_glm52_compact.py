#!/usr/bin/env python3
"""Save one serving run, or compare two saved greedy runs. Standard library only.

Does not set forced budgets, flush caches, or launch servers. Non-streaming
latency is end-to-end latency, not TTFT/TPOT. Telemetry is supporting evidence,
not proof of NPU graph replay or TP correctness.
"""

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "请详细解释二分查找，给出 Python 实现、边界情况和复杂度分析。",
    "Explain speculative decoding and why a target model must verify draft tokens.",
    "写一篇关于城市雨夜的短篇故事，包含开头、转折和结尾。",
    "推导等差数列的求和公式，并用三个例子验证。",
    "Write a Python LRU cache and explain how each method works.",
    "请比较数据库的乐观锁与悲观锁，给出具体使用场景。",
    "Summarize this report in detail:\n"
    + "The team measured latency and throughput under increasing load. " * 128,
    "用通俗语言详细介绍张量并行中的 all-reduce 和 all-gather。",
]


def http_json(url, timeout, payload=None):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def status(args):
    info = http_json(args.base_url.rstrip("/") + "/server_info", args.timeout)
    fields = (
        "speculative_algorithm",
        "model_path",
        "served_model_name",
        "tp_size",
        "nnodes",
        "disable_overlap_schedule",
        "cuda_graph_config",
        "speculative_dspark_sps_table_path",
    )
    states = []
    for index, state in enumerate(info.get("internal_states") or []):
        payload = state.get("dspark_info_record")
        record = payload or {}
        states.append(
            {
                "state_index": index,
                **{key: state.get(key, info.get(key)) for key in fields},
                "token_capacity": state.get("memory_usage", {}).get("token_capacity"),
                "recording_available": payload is not None,
                "ragged_mode": record.get("mode"),
                "simulate_acc_len": record.get("simulate_acc_len"),
                "record_components": record.get("components"),
                "record_count": len(record.get("records", [])),
            }
        )
    print(
        json.dumps(
            {
                "base_url": args.base_url,
                "server": {key: info.get(key) for key in fields},
                "internal_states": states,
                "note": "No record payload means ragged mode/simulation is unknown here; null fields alone do not prove compact is enabled.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def generate(base, prompt, max_tokens, timeout):
    started = time.monotonic()
    try:
        result = http_json(
            base + "/generate",
            timeout,
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
                "stream": False,
            },
        )
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError("Expected /generate response with text")
        return {"ok": True, "seconds": time.monotonic() - started, "response": result}
    except Exception as error:
        return {"ok": False, "seconds": time.monotonic() - started, "error": str(error)}


def telemetry(before, after):
    old_states = before.get("internal_states") or []
    summaries = []
    for index, state in enumerate(after.get("internal_states") or []):
        payload = state.get("dspark_info_record")
        if payload is None:
            summaries.append({"dp_index": index, "recording_available": False})
            continue
        old_payload = (
            (old_states[index].get("dspark_info_record") or {})
            if index < len(old_states)
            else {}
        )
        watermark = max(
            (r.get("forward_ct", -1) for r in old_payload.get("records", [])),
            default=-1,
        )
        rows = [
            r for r in payload.get("records", []) if r.get("forward_ct", -1) > watermark
        ]
        width = payload.get("verify_num_draft_tokens", 0)
        smaller = [
            r
            for r in rows
            if r.get("bs", 0) > 0
            and width > 0
            and 0 < r.get("verify_tokens_graph_key", -1) < r["bs"] * width
        ]
        summaries.append(
            {
                "dp_index": index,
                "recording_available": True,
                "mode": payload.get("mode"),
                "simulate_acc_len": payload.get("simulate_acc_len"),
                "verify_num_draft_tokens": width,
                "new_records": len(rows),
                "records_with_budget": sum(r.get("budget") is not None for r in rows),
                "distinct_budgets": sorted(
                    {r["budget"] for r in rows if r.get("budget") is not None}
                ),
                "graph_token_keys": sorted(
                    {
                        r["verify_tokens_graph_key"]
                        for r in rows
                        if r.get("verify_tokens_graph_key", -1) > 0
                    }
                ),
                "records_with_graph_key_below_full_width": len(smaller),
                "examples": smaller[:5],
            }
        )
    return summaries


def capture(args):
    base = args.base_url.rstrip("/")
    before = http_json(base + "/server_info", args.timeout)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(generate, base, p, args.max_tokens, args.timeout)
            for p in prompts
        ]
        results = [f.result() for f in futures]
    wall = time.monotonic() - started
    after = http_json(base + "/server_info", args.timeout)
    successful = [r for r in results if r["ok"]]
    latencies = sorted(r["seconds"] for r in successful)
    counts = [
        r["response"].get("meta_info", {}).get("completion_tokens") for r in successful
    ]
    tokens = sum(counts) if counts and all(isinstance(n, int) for n in counts) else None
    summary = {
        "succeeded": len(successful),
        "requested": len(results),
        "wall_seconds": wall,
        "completion_tokens": tokens,
        "observed_tokens_per_second": tokens / wall if tokens is not None else None,
        "p95_successful_request_seconds": latencies[
            math.ceil(0.95 * len(latencies)) - 1
        ]
        if latencies
        else None,
        "dspark_telemetry": telemetry(before, after),
    }
    report = {
        "settings": {
            "base_url": base,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
        "prompts": prompts,
        "results": results,
        "summary": summary,
        "server_info_before": before,
        "server_info_after": after,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", args.out)
    return 0 if len(successful) == len(results) else 1


def compare(args):
    baseline, candidate = [
        json.loads(p.read_text()) for p in (args.baseline, args.candidate)
    ]
    for field in ("max_tokens", "temperature", "requests", "concurrency"):
        if baseline["settings"][field] != candidate["settings"][field]:
            raise ValueError("Different workload setting: " + field)
    if baseline["prompts"] != candidate["prompts"]:
        raise ValueError("Different prompts")
    if len(baseline["results"]) != len(candidate["results"]) or not baseline["results"]:
        raise ValueError("Empty or mismatched results")
    mismatches = []
    for i, (left, right) in enumerate(zip(baseline["results"], candidate["results"])):
        if not (left["ok"] and right["ok"]):
            mismatches.append({"index": i, "reason": "request failed"})
        elif left["response"]["text"] != right["response"]["text"]:
            mismatches.append({"index": i, "reason": "text differs"})
        elif (
            left["response"].get("output_ids") is not None
            and right["response"].get("output_ids") is not None
            and left["response"]["output_ids"] != right["response"]["output_ids"]
        ):
            mismatches.append({"index": i, "reason": "output token IDs differ"})
    print(
        json.dumps(
            {"total": len(baseline["results"]), "mismatches": mismatches}, indent=2
        )
    )
    return 1 if mismatches else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("capture")
    run.add_argument("--base-url", default="http://127.0.0.1:8030")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--requests", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--out", type=Path, required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    inspect = sub.add_parser(
        "status", help="Read compact configuration without sending generation requests"
    )
    inspect.add_argument("--base-url", default="http://127.0.0.1:8030")
    inspect.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.command == "status":
        return status(args)
    if args.command == "capture":
        if min(args.concurrency, args.requests, args.max_tokens, args.timeout) <= 0:
            parser.error("Workload counts and timeout must be positive")
        return capture(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
