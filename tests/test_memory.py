from __future__ import annotations

from research import memory


def test_summarise_formats_chunks_for_llm(monkeypatch) -> None:
    seen = {}

    def fake_complete(messages, *, system=None, max_tokens=None):
        seen["messages"] = messages
        seen["system"] = system
        return "answer"

    monkeypatch.setattr(memory, "complete", fake_complete)

    answer = memory.summarise(
        [{"source": "doc.pdf", "text": "alpha fact"}],
        "What is alpha?",
    )

    assert answer == "answer"
    assert "doc.pdf" in seen["messages"][0]["content"]
    assert "alpha fact" in seen["messages"][0]["content"]
    assert "provided chunks" in seen["system"]


def test_summarise_handles_empty_chunks_without_llm() -> None:
    assert memory.summarise([], "question") == (
        "No relevant information found in the knowledge base."
    )


def test_update_adds_new_info_when_no_existing_chunks(monkeypatch) -> None:
    added = []

    monkeypatch.setattr(memory, "search", lambda topic, top_k=None: [])
    monkeypatch.setattr(memory, "add", added.append)

    assert memory.update("alpha", "new alpha fact") == "new alpha fact"
    assert added == ["new alpha fact"]


def test_update_merges_deletes_dominant_source_and_reingests(monkeypatch) -> None:
    deleted_sources = []
    added = []

    existing_chunks = [
        {"source": "a.txt", "text": "old alpha 1"},
        {"source": "a.txt", "text": "old alpha 2"},
        {"source": "b.txt", "text": "related beta"},
    ]

    class FakeConnection:
        def execute(self, sql, params=()):
            deleted_sources.append(params[0])

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(memory, "search", lambda topic, top_k=None: existing_chunks)
    monkeypatch.setattr(memory, "complete", lambda messages, *, system=None: "merged note")
    monkeypatch.setattr(memory, "init_db", lambda: FakeConnection())
    monkeypatch.setattr(memory, "add", added.append)

    assert memory.update("alpha", "new alpha") == "merged note"
    assert deleted_sources == ["a.txt"]
    assert added == ["merged note"]
