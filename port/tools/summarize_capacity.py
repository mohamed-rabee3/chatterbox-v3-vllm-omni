#!/usr/bin/env python3
"""Summarize TTS-only Locust capacity runs, keeping failed runs visible."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-start-s", type=float, default=1.0)
    parser.add_argument("--max-stall-s", type=float, default=0.1)
    args = parser.parse_args()
    print("| Run | Users | Completed | Errors / unfinished | TTFA p95 | Playback start p95 | Buffered stall p95 | Turns with >100ms stall | Meets criteria |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    reports = []
    for path in args.root.rglob("locust_conversation.json"):
        report = json.loads(path.read_text())
        if "buffered_playback_stall_s_p95" in report:
            reports.append((path, report))
    for path, d in sorted(reports, key=lambda pair: (pair[1]["users"], str(pair[0]))):
        good = (d["turns_completed"] > 0 and d["failures"] == 0
                and d["requests_unfinished"] == 0
                and d["playback_start_s_p95"] <= args.max_start_s
                and d["buffered_playback_stall_s_p95"] <= args.max_stall_s)
        print(f"| {path.parent.relative_to(args.root)} | {d['users']} | {d['turns_completed']} | "
              f"{d['failures']} / {d['requests_unfinished']} | {d['ttfa_s']['p95']:.3f}s | "
              f"{d['playback_start_s_p95']:.3f}s | {d['buffered_playback_stall_s_p95']:.3f}s | "
              f"{d['buffered_turns_stalled_over_100ms_pct']:.1f}% | {'yes' if good else 'no'} |")
    print(f"\nCriteria: p95 playback start <= {args.max_start_s}s, p95 cumulative buffered "
          f"stall <= {args.max_stall_s}s, zero errors/unfinished requests. "
          "See each report for its client buffer, pacing, sample size, and workload. "
          "Passing a short sample is not production qualification.")


if __name__ == "__main__":
    main()
