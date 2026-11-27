import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import connect_claude_desktop as c  # noqa: E402


def test_config_path_per_os():
    home = Path("/home/u")
    assert c.config_path("Darwin", {}, home) == home / "Library/Application Support/Claude/claude_desktop_config.json"
    assert c.config_path("Linux", {}, home) == home / ".config/Claude/claude_desktop_config.json"
    win = c.config_path("Windows", {"APPDATA": "/appdata"}, home)
    assert win == Path("/appdata/Claude/claude_desktop_config.json")


def test_entry_has_no_cwd_and_no_secrets():
    entry = c.build_entry("/usr/bin/python3", "hash", "/data/hash")
    assert entry == {"command": "/usr/bin/python3", "args": ["-m", "app.mcp_server"],
                     "env": {"EMBEDDER": "hash", "QDRANT_PATH": "/data/hash"}}
    assert "cwd" not in entry
    assert not any("key" in k.lower() or "token" in k.lower() for k in entry["env"])


def test_merge_keeps_other_servers_and_settings():
    existing = json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}})
    out = json.loads(c.merge_config(existing, c.build_entry("py", "hash", "/d")))
    assert out["theme"] == "dark" and out["mcpServers"]["other"] == {"command": "x"}
    assert out["mcpServers"]["memory-assistant"]["command"] == "py"


def test_merge_into_an_empty_or_new_file():
    assert json.loads(c.merge_config("", c.build_entry("py", "hash", "/d")))["mcpServers"]["memory-assistant"]
    assert json.loads(c.merge_config("   \n", c.build_entry("py", "hash", "/d")))["mcpServers"]["memory-assistant"]


def test_merge_replaces_an_old_entry_and_remove_deletes_only_ours():
    first = c.merge_config(json.dumps({"mcpServers": {"other": {"command": "x"}}}), c.build_entry("old", "hash", "/d"))
    second = json.loads(c.merge_config(first, c.build_entry("new", "fastembed", "/d2")))
    assert second["mcpServers"]["memory-assistant"]["command"] == "new"
    removed = json.loads(c.merge_config(first, None))
    assert list(removed["mcpServers"]) == ["other"]


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '{"mcpServers": []}'])
def test_merge_refuses_files_it_does_not_understand(bad):
    with pytest.raises(ValueError):
        c.merge_config(bad, c.build_entry("py", "hash", "/d"))


def test_write_makes_a_backup_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "Claude" / "claude_desktop_config.json"
    assert c.write_with_backup(path, "one\n") is None  # nothing to back up the first time
    backup = c.write_with_backup(path, "two\n", now=datetime(2026, 11, 28, 9, 0, 0))
    assert backup.name == "claude_desktop_config.json.bak-20261128090000"
    assert backup.read_text() == "one\n" and path.read_text() == "two\n"
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([path.name, backup.name])


def test_main_dry_run_writes_nothing_and_a_bad_file_is_left_alone(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{broken")
    assert c.main(["--config", str(cfg), "--skip-check"]) == 1
    assert cfg.read_text() == "{broken"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    assert c.main(["--config", str(cfg), "--skip-check", "--dry-run"]) == 0
    assert json.loads(cfg.read_text()) == {"mcpServers": {}}
    assert "memory-assistant" in capsys.readouterr().out


def test_main_writes_then_removes(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    assert c.main(["--config", str(cfg), "--skip-check", "--data-dir", str(tmp_path / "mem")]) == 0
    written = json.loads(cfg.read_text())["mcpServers"]
    assert written["memory-assistant"]["env"]["QDRANT_PATH"] == str(tmp_path / "mem") and "other" in written
    assert c.main(["--config", str(cfg), "--skip-check", "--remove"]) == 0
    assert list(json.loads(cfg.read_text())["mcpServers"]) == ["other"]
    assert len(list(tmp_path.glob("cfg.json.bak-*"))) >= 1


def test_the_server_check_really_starts_the_server_and_round_trips():
    pytest.importorskip("mcp")
    assert c.check_server(sys.executable, "hash") == []


def test_the_server_check_reports_a_server_that_cannot_start():
    problems = c.check_server(sys.executable + "-does-not-exist", "hash")
    assert problems and "could not start" in problems[0]
