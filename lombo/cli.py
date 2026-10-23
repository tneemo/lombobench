"""LomboBench CLI: `python -m lombo`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .proxy import LomboConfig, create_app

PRIVACY_NOTICE = (
    "NOTICE: LomboProxy records intercepted prompts and responses in a local database.\n"
    "Conversations may contain sensitive data. Use X-Lombo-No-Log to exclude a\n"
    "request from logging, or --no-capture to disable capture."
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lombo",
        description="LomboBench — transparent proxy + Lab for observing "
                    "Ollama's external inference controls.")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1",
                        help="for privacy, listens only on localhost by default")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--db", default="lombo.db")
    parser.add_argument("--no-capture", action="store_true",
                        help="forward requests without recording anything")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.host != "127.0.0.1":
        print(f"WARNING: listening on {args.host} — the proxy is accessible outside this "
              f"machine and captured conversations may contain sensitive data.")
    print(PRIVACY_NOTICE if not args.no_capture else "Capture DISABLED (--no-capture).")
    print(f"\n  Proxy (Ollama drop-in): http://{args.host}:{args.port}")
    print(f"  Lab / analysis:             http://{args.host}:{args.port}/_lombo/")
    print(f"  Ollama upstream:           {args.ollama_url}")
    print(f"  Database:                     {args.db}\n")

    config = LomboConfig(ollama_url=args.ollama_url, db_path=args.db,
                         capture=not args.no_capture)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
