"""Measure LomboProxy overhead: N identical direct × proxied calls.

The proxy streams transparently, but transparency is not free — this script
measures the actual cost (median total latency and time-to-first-token) so
that the number reported in the paper is observed rather than claimed.

Usage: python scripts/overhead_bench.py [--n 8] [--model llama3.2:3b]
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx


def bench(base_url: str, model: str, n: int, extra_headers: dict) -> tuple[float, float]:
    totals, ttfts = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        ttft = None
        with httpx.stream("POST", f"{base_url}/api/generate", json={
                "model": model, "stream": True,
                "options": {"seed": 1, "temperature": 0, "num_predict": 30},
                "prompt": "Count from 1 to 5."},
                headers=extra_headers, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_raw():
                if chunk and ttft is None:
                    ttft = time.perf_counter() - t0
        totals.append(time.perf_counter() - t0)
        ttfts.append(ttft)
    return statistics.median(totals) * 1000, statistics.median(ttfts) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--direct", default="http://127.0.0.1:11434")
    parser.add_argument("--proxy", default="http://127.0.0.1:11435")
    args = parser.parse_args()

    # warm up the model to exclude loading from the first measured call
    httpx.post(f"{args.direct}/api/generate", json={
        "model": args.model, "prompt": "hi", "stream": False,
        "options": {"num_predict": 5}}, timeout=300)

    d_total, d_ttft = bench(args.direct, args.model, args.n, {})
    p_total, p_ttft = bench(args.proxy, args.model, args.n,
                            {"X-Lombo-Experiment": "overhead-bench"})
    print(f"n = {args.n} calls per path | model {args.model}")
    print(f"direct       : median total {d_total:8.1f} ms | median TTFT {d_ttft:7.1f} ms")
    print(f"proxied      : median total {p_total:8.1f} ms | median TTFT {p_ttft:7.1f} ms")
    print(f"overhead     : total {p_total - d_total:+8.1f} ms | TTFT {p_ttft - d_ttft:+7.1f} ms")


if __name__ == "__main__":
    main()
