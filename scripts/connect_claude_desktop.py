"""Connect the Memory Assistant MCP server to Claude Desktop with one command.

    pip install -e ".[mcp]"                      # once, from the repo folder
    python scripts/connect_claude_desktop.py     # checks the server, then edits Claude Desktop's config
    python scripts/connect_claude_desktop.py --dry-run     # show what it would write, change nothing
    python scripts/connect_claude_desktop.py --remove      # take the entry out again

What it does, in order:
  1. Starts the server exactly the way Claude Desktop will (this Python, `-m app.mcp_server`, from a
     different folder) and calls remember / recall / forget once. If that fails, it stops and writes nothing.
  2. Adds a "memory-assistant" entry to claude_desktop_config.json, keeping every other server, after
     saving a timestamped backup. If the file is not valid JSON it refuses to touch it.
  3. Memory is stored on disk in ~/.memory-assistant/<embedder>, so it survives restarting Claude Desktop.

It writes no API key or secret anywhere. Restart Claude Desktop completely afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

SERVER_NAME = "memory-assistant"


def config_path(system: str, env: dict, home: Path) -> Path:
    """Where Claude Desktop keeps its config on each OS."""
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def build_entry(python: str, embedder: str, data_dir: str) -> dict:
    """The config entry. No `cwd` (not a documented field) and no secrets: the package is pip-installed."""
    return {
        "command": python,
        "args": ["-m", "app.mcp_server"],
        "env": {"EMBEDDER": embedder, "QDRANT_PATH": data_dir},
    }


def merge_config(existing_text: str, entry: Optional[dict]) -> str:
    """Return the new file text with our entry set (or removed when entry is None). Other servers stay."""
    try:
        data = json.loads(existing_text) if existing_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"the config file is not valid JSON ({exc}); not touching it") from exc
    if not isinstance(data, dict):
        raise ValueError("the config file is not a JSON object; not touching it")
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError('"mcpServers" in the config is not an object; not touching it')
    if entry is None:
        servers.pop(SERVER_NAME, None)
    else:
        servers[SERVER_NAME] = entry
    data["mcpServers"] = servers
    return json.dumps(data, indent=2) + "\n"


def write_with_backup(path: Path, text: str, now: Optional[datetime] = None) -> Optional[Path]:
    """Write `text` to `path`, first copying any existing file to a timestamped .bak. Returns the backup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return backup


async def _round_trip(python: str, env: dict, cwd: str) -> list:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=python, args=["-m", "app.mcp_server"], env=env, cwd=cwd)
    problems = []
    user = f"check-{uuid.uuid4().hex[:8]}"
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            if tools != {"remember", "recall", "forget"}:
                problems.append(f"unexpected tools: {sorted(tools)}")
            stored = json.loads((await session.call_tool("remember", {"user_id": user, "text": "I live in Pune.", "key": "lives_in"})).content[0].text)
            found = json.loads((await session.call_tool("recall", {"user_id": user, "query": "where do I live"})).content[0].text)
            if not any("Pune" in m["text"] for m in found.get("memories", [])):
                problems.append("recall did not return the stored fact")
            gone = json.loads((await session.call_tool("forget", {"user_id": user, "memory_id": stored["memory"]["id"]})).content[0].text)
            if gone != {"deleted": True}:
                problems.append(f"forget returned {gone}")
    return problems


def check_server(python: str, embedder: str) -> list:
    """Start the server the way Claude Desktop will and use it once. Returns a list of problems (empty = ok).

    It runs from a temporary folder, so it only works if the package is properly pip-installed, which is
    exactly what Claude Desktop needs. It uses a throwaway storage folder, never your real memory.
    """
    with tempfile.TemporaryDirectory() as scratch:
        env = dict(os.environ, EMBEDDER=embedder, QDRANT_PATH=str(Path(scratch) / "mem"))
        for name in ("QDRANT_URL", "MCP_API_KEY", "MCP_USER_ID"):
            env.pop(name, None)
        try:
            return asyncio.run(_round_trip(python, env, scratch))
        except Exception as exc:  # report, do not crash: the caller prints the fix
            return [f"could not start or use the server: {type(exc).__name__}: {exc}"]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=["hash", "fastembed"], default="hash",
                        help="hash: instant, word-overlap only. fastembed: real model, downloads ~130 MB on first use")
    parser.add_argument("--data-dir", default=None, help="where memory is kept (default ~/.memory-assistant/<embedder>)")
    parser.add_argument("--config", type=Path, default=None, help="path to claude_desktop_config.json (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    parser.add_argument("--remove", action="store_true", help="remove the memory-assistant entry")
    parser.add_argument("--skip-check", action="store_true", help="do not start the server first")
    args = parser.parse_args(argv)

    path = args.config or config_path(platform.system(), dict(os.environ), Path.home())
    data_dir = args.data_dir or str(Path.home() / ".memory-assistant" / args.embedder)
    python = sys.executable
    print(f"Claude Desktop config: {path}")

    if args.remove:
        entry = None
    else:
        if not args.skip_check:
            print("Checking the server starts and works (remember / recall / forget)...")
            problems = check_server(python, args.embedder)
            if problems:
                print("The server check FAILED, so nothing was written:")
                for p in problems:
                    print(f"  - {p}")
                print('Fix: from the repo folder run   pip install -e ".[mcp]"   with this same Python:')
                print(f"  {python}")
                return 1
            print("  ok")
        entry = build_entry(python, args.embedder, data_dir)

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        new_text = merge_config(existing, entry)
    except ValueError as exc:
        print(f"Stopped: {exc}")
        return 1

    if args.dry_run:
        print("\n--dry-run: would write\n" + new_text)
        return 0
    backup = write_with_backup(path, new_text)
    if backup:
        print(f"Backup of the old config: {backup}")
    print("Removed the entry." if args.remove else f"Added '{SERVER_NAME}'. Memory is stored in {data_dir}")
    print("\nNext: quit Claude Desktop completely (including the tray icon) and reopen it.")
    print("Then check the tools menu for remember / recall / forget, and try:")
    print('  "Remember that my favourite language is Python."  ->  new chat  ->  "What is my favourite language?"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
