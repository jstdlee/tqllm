#!/usr/bin/env python3
"""Single-request benchmark against a tqLLM server. Prints server timing fields."""
import json, sys, time, urllib.request

url = sys.argv[1]
label = sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
warmup = len(sys.argv) > 4 and sys.argv[4] == "--warmup"

PROMPT = (
    "Implement an O(1) get/put LRU cache in Python supporting capacity eviction. "
    "Think through the data-structure choices, the invariants you must keep, and "
    "the edge cases (capacity 0/1, evict-on-put, touch-on-get). Then give the "
    "final code with a 3-line proof of the amortized O(1) claim."
)

def bench(url, label):
    body = {"model": "tq27b", "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.loads(r.read())
    wall = time.time() - t0
    u = resp.get("usage", {})
    t = resp.get("timings", {})
    c = resp["choices"][0]["message"]
    out = {
        "label": label, "wall_s": round(wall, 2), "finish": resp["choices"][0]["finish_reason"],
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "answer_len": len(c.get("content") or ""), "think_len": len(c.get("reasoning_content") or ""),
        "decode_tok_s": t.get("predicted_per_second"), "prefill_tok_s": t.get("prompt_per_second"),
        "predict_ms": t.get("predicted_ms"),
        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
        "eff_tok_s": round(u.get("completion_tokens", 0) / wall, 1) if u.get("completion_tokens") else None,
    }
    print(json.dumps(out, indent=1))
    return out

if __name__ == "__main__":
    if warmup:
        print("## warmup", file=sys.stderr)
        bench(url, label + "-warmup")
    bench(url, label)
