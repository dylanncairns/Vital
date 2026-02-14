from __future__ import annotations

import argparse
import time

from api.main import ProcessJobsIn, process_background_jobs


def run_once(*, limit: int = 20, max_papers_per_query: int = 8) -> dict:
    return process_background_jobs(
        ProcessJobsIn(limit=limit, max_papers_per_query=max_papers_per_query)
    )


def run_forever(
    *,
    limit: int = 20,
    max_papers_per_query: int = 8,
    interval_seconds: float = 2.0,
    idle_sleep_seconds: float = 5.0,
) -> None:
    while True:
        result = run_once(limit=limit, max_papers_per_query=max_papers_per_query)
        jobs_done = int(result.get("jobs_done", 0))
        jobs_claimed = int(result.get("jobs_claimed", 0))
        print(result)
        if jobs_claimed == 0:
            time.sleep(idle_sleep_seconds)
        else:
            time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run background jobs worker for insights/evidence.")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    parser.add_argument("--limit", type=int, default=20, help="Max jobs per batch")
    parser.add_argument("--max-papers-per-query", type=int, default=8, help="RAG retrieval rows per candidate")
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Sleep between active batches")
    parser.add_argument("--idle-sleep-seconds", type=float, default=5.0, help="Sleep when no jobs are pending")
    args = parser.parse_args()

    if args.once:
        print(run_once(limit=args.limit, max_papers_per_query=args.max_papers_per_query))
        return

    run_forever(
        limit=args.limit,
        max_papers_per_query=args.max_papers_per_query,
        interval_seconds=args.interval_seconds,
        idle_sleep_seconds=args.idle_sleep_seconds,
    )


if __name__ == "__main__":
    main()
