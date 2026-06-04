"""
cli.py — command-line interface.

Three commands:

    add <source>                 ingest a PDF, URL, or text note
    ask <question>               retrieve and synthesise an answer
    update <topic> <new_info>    merge new information into existing knowledge

Usage (via main.py / uv run):

    uv run main.py add path/to/nasa_jpl.pdf
    uv run main.py add https://nasa.gov/some-doc
    uv run main.py add "Rule 1: no recursion allowed"

    uv run main.py ask "what does NASA say about recursion?"

    uv run main.py update "recursion" "JPL 2024: recursion is safety-critical"
"""

from __future__ import annotations

import argparse
import sys
import textwrap


def _print_answer(answer: str) -> None:
    """Print a wrapped answer with a divider."""
    print()
    print("─" * 72)
    print(
        textwrap.fill(answer, width=72, break_long_words=False, break_on_hyphens=False)
    )
    print("─" * 72)
    print()


def _print_chunks_header(chunks: list[dict]) -> None:
    """Print a one-line summary of where the answer is drawn from."""
    if not chunks:
        return
    sources = sorted({c["source"] for c in chunks})
    short = [s.split("/")[-1] if "/" in s else s for s in sources]
    print(f"  sources: {', '.join(short)}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    from research.ingest import add, _detect_source_type

    source = args.source
    source_type = _detect_source_type(source)

    type_label = {"pdf": "PDF", "url": "URL", "text": "note"}.get(
        source_type, source_type
    )
    print(f"Adding {type_label}: {source!r}")
    print("  parsing and chunking ...", end="", flush=True)

    try:
        add(source)
    except ValueError as e:
        print()
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print()
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    print(" done")
    print("  embedding and storing ...", end="", flush=True)
    # (embedding happens inside add(); the progress message is cosmetic —
    #  by the time we reach this line it's already complete)
    print(" done")
    print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from research.retrieve import search
    from research.memory import summarise

    question = args.question
    print(f"Searching for: {question!r}")

    try:
        chunks = search(question)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Search error: {e}", file=sys.stderr)
        return 1

    if not chunks:
        print("Nothing found in the knowledge base yet.")
        print("Try `add`-ing some sources first.")
        return 0

    _print_chunks_header(chunks)
    print("  synthesising ...", end="", flush=True)

    try:
        answer = summarise(chunks, question)
    except Exception as e:
        print()
        print(f"Synthesis error: {e}", file=sys.stderr)
        return 1

    print(" done")
    _print_answer(answer)
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    from research.memory import update

    topic = args.topic
    new_info = args.new_info

    print(f"Updating knowledge on: {topic!r}")
    print("  retrieving existing knowledge ...", end="", flush=True)

    try:
        merged = update(topic, new_info)
    except ValueError as e:
        print()
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print()
        print(f"Update error: {e}", file=sys.stderr)
        return 1

    print(" done")
    print()
    print("Stored note:")
    _print_answer(merged)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="A lightweight personal research system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              uv run main.py add path/to/nasa_jpl.pdf
              uv run main.py add https://nasa.gov/some-doc
              uv run main.py add "Rule 1: no recursion allowed"
              uv run main.py ask "what does NASA say about recursion?"
              uv run main.py update "recursion" "JPL 2024: safety-critical"
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # add
    p_add = sub.add_parser("add", help="ingest a PDF, URL, or text note")
    p_add.add_argument(
        "source",
        help="PDF file path, https:// URL, or raw text string",
    )

    # ask
    p_ask = sub.add_parser("ask", help="ask a question against your knowledge base")
    p_ask.add_argument("question", help="natural-language question")

    # update
    p_update = sub.add_parser(
        "update", help="merge new information into existing knowledge"
    )
    p_update.add_argument("topic", help="short description of the topic")
    p_update.add_argument("new_info", help="the new information to incorporate")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "add":
        return cmd_add(args)
    elif args.command == "ask":
        return cmd_ask(args)
    elif args.command == "update":
        return cmd_update(args)
    else:
        parser.print_help()
        return 1
