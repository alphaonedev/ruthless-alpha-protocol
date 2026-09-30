"""Command-line runner for a JSON watchlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alpaca import daily_bars
from .protocol import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ruthless Alpha Protocol")
    parser.add_argument("watchlist", type=Path, help="JSON array of symbol/yield/kill_probability objects")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    items = json.loads(args.watchlist.read_text(encoding="utf-8"))
    symbols = [item["symbol"].upper() for item in items]
    bars = daily_bars(symbols)
    report = [analyze(s, bars[s], float(i["cash_yield"]), float(i["kill_probability"]))
              for i, s in zip(items, symbols)]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

