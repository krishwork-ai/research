from __future__ import annotations

from research import cli


def test_main_routes_add_command(monkeypatch, capsys) -> None:
    import research.ingest

    calls = []
    monkeypatch.setattr(research.ingest, "add", calls.append)
    monkeypatch.setattr(research.ingest, "_detect_source_type", lambda source: "text")

    assert cli.main(["add", "alpha note"]) == 0

    captured = capsys.readouterr()
    assert calls == ["alpha note"]
    assert "Adding note" in captured.out


def test_main_routes_ask_command(monkeypatch, capsys) -> None:
    import research.memory
    import research.retrieve

    chunks = [{"source": "doc.txt", "text": "alpha fact"}]
    monkeypatch.setattr(research.retrieve, "search", lambda question: chunks)
    monkeypatch.setattr(
        research.memory,
        "summarise",
        lambda retrieved, question: f"answer from {retrieved[0]['source']}",
    )

    assert cli.main(["ask", "alpha?"]) == 0

    captured = capsys.readouterr()
    assert "sources: doc.txt" in captured.out
    assert "answer from doc.txt" in captured.out


def test_main_routes_update_command(monkeypatch, capsys) -> None:
    import research.memory

    monkeypatch.setattr(research.memory, "update", lambda topic, info: "merged note")

    assert cli.main(["update", "alpha", "new fact"]) == 0

    captured = capsys.readouterr()
    assert "Stored note:" in captured.out
    assert "merged note" in captured.out
