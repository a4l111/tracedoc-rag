from __future__ import annotations

import argparse

import uvicorn

from src.service.web import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the TraceDoc RAG LAN web interface.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use a LAN IP only on a trusted network.")
    parser.add_argument("--port", default=8000, type=int, help="TCP port, from 1 to 65535.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
