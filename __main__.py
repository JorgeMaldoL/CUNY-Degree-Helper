"""
Module entry point for `python -m CUNY_Degree_Helper`.

Provides utilities:
- `reindex` to (re)build the ChromaDB collection.
- `app` to run the Streamlit app (for local dev convenience).

Usage:
    python -m CUNY_Degree_Helper reindex [--force]
    python -m CUNY_Degree_Helper app
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_reindex(force: bool = False) -> int:
    # Ensure backend is importable
    sys.path.append(str(Path(__file__).parent / "backend"))
    from backend.chroma_manager import ChromaDBManager

    mgr = ChromaDBManager()
    count = mgr.reindex(force=force)
    print(f"Indexed {count} programs.")
    return 0


def cmd_app() -> int:
    # Run Streamlit app for local convenience
    repo = Path(__file__).parent
    entry = repo / "frontend" / "frontend.py"
    if not entry.exists():
        print("frontend/frontend.py not found.")
        return 1

    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry)]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="CUNY_Degree_Helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("reindex", help="(re)build the ChromaDB collection")
    p1.add_argument("--force", action="store_true", help="Drop and rebuild the collection")

    sub.add_parser("app", help="Run the Streamlit app")

    args = parser.parse_args(argv)
    if args.cmd == "reindex":
        return cmd_reindex(force=args.force)
    if args.cmd == "app":
        return cmd_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
