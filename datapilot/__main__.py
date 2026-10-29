"""Command-line interface.

    python -m datapilot data.csv --target churn
    python -m datapilot data.csv --ask "average tenure by contract"
    python -m datapilot data.csv --chat
"""
from __future__ import annotations

import argparse

from .graph import DataPilot


def main() -> None:
    p = argparse.ArgumentParser(prog="datapilot", description="Autonomous data scientist for CSV files")
    p.add_argument("csv")
    p.add_argument("--target", default="auto", help='column to predict, "auto", or "none"')
    p.add_argument("--clip-outliers", action="store_true")
    p.add_argument("--ask", action="append", default=[], help="question for the assistant (repeatable)")
    p.add_argument("--chat", action="store_true", help="interactive Q&A after the analysis")
    a = p.parse_args()

    target = None if a.target.lower() == "none" else a.target
    pilot = DataPilot(a.csv, target, clip_outliers=a.clip_outliers)
    s = pilot.state
    print("\n".join(f"[graph] {t}" for t in s["trace"]))
    print("\n== Cleaning ==\n" + s["cleaning_log"])
    if s.get("ml") is not None:
        print("\n== Models ==\n" + s["ml"].summary())
    elif s.get("ml_error"):
        print(f"\n== Models ==\nFailed: {s['ml_error']}")
    print("\n== Insights ==\n" + s["insights"])

    def show(q: str) -> None:
        out = pilot.ask(q)
        print(f"\n[{out['route']}] Q: {q}")
        if out.get("sql") is not None and out["sql"].sql:
            print("SQL:", out["sql"].sql)
            if out["sql"].result is not None:
                print(out["sql"].result.head(15).to_string(index=False))
        print("A:", out["answer"])

    for q in a.ask:
        show(q)
    if a.chat:
        while True:
            try:
                q = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("", "exit", "quit"):
                break
            show(q)


if __name__ == "__main__":
    main()
