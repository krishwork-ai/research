"""
memory.py — synthesis and memory update layer.

Two public functions:

    summarise(chunks)              -> str
        LLM call: synthesise retrieved chunks into a cited answer.
        Used by the `ask` command.

    update(topic, new_info)        -> str
        Retrieve existing knowledge on a topic, merge with new_info
        via LLM, replace the stored entry with the merged result.
        Returns the merged text so the CLI can confirm what was stored.

Update strategy
---------------
Rather than appending new chunks alongside old ones (which pollutes
retrieval with duplicates), update():

  1. Retrieves the top-k chunks most relevant to `topic`.
  2. Calls the LLM to merge existing knowledge + new_info into a single
     coherent note.
  3. Deletes the source entry that contributed the most retrieved chunks
     (cascade removes its chunks + embeddings).
  4. Re-ingests the merged note as a new 'text' entry.

This keeps the store clean — one entry per distinct knowledge unit,
updated in place rather than accumulated.
"""

from __future__ import annotations

from collections import Counter

from research.config import config
from research.llm import complete
from research.retrieve import search
from research.store import init_db
from research.ingest import add


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARISE_SYSTEM = """\
You are a precise technical assistant helping a researcher understand \
NASA and aerospace engineering standards.

You will be given a set of retrieved text chunks and a question. \
Synthesise a clear, accurate answer using only the provided chunks. \
Cite your sources inline using the format [source: <filename or url>]. \
If the chunks do not contain enough information to answer, say so clearly \
rather than guessing.\
"""

_MERGE_SYSTEM = """\
You are a precise technical note-taking assistant. \
Your job is to merge an existing knowledge note with new information \
into a single, coherent, non-redundant note. \
Preserve all distinct facts from both sources. \
Remove any information directly contradicted by the new information. \
Write in clear, concise prose. Do not add commentary or preamble — \
output only the merged note text.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarise(chunks: list[dict], question: str) -> str:
    """
    Synthesise retrieved chunks into a cited answer.

    Parameters
    ----------
    chunks:
        List of chunk dicts as returned by retrieve.search() — each must
        have at least 'text' and 'source' keys.
    question:
        The original user question, used to focus the synthesis.

    Returns
    -------
    str
        A coherent answer with inline source citations.
        If chunks is empty, returns a message saying nothing was found.
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    # Format chunks for the LLM context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[{i}] Source: {source}\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "user",
            "content": (f"Question: {question}\n\nRetrieved chunks:\n\n{context}"),
        }
    ]

    return complete(messages, system=_SUMMARISE_SYSTEM)


def update(topic: str, new_info: str) -> str:
    """
    Merge new information about a topic into the knowledge base.

    Retrieves the most relevant existing chunks for `topic`, merges them
    with `new_info` using the LLM, removes the dominant source entry,
    and re-ingests the merged result.

    Parameters
    ----------
    topic:
        A short description of what the new information is about.
        Used as the retrieval query to find related existing chunks.
    new_info:
        The new information to incorporate.

    Returns
    -------
    str
        The merged note text that was stored, so the caller can confirm.

    Raises
    ------
    ValueError
        If topic or new_info is empty.
    """
    topic = topic.strip()
    new_info = new_info.strip()

    if not topic:
        raise ValueError("topic must not be empty")
    if not new_info:
        raise ValueError("new_info must not be empty")

    # Retrieve the most relevant existing chunks for this topic
    existing_chunks = search(topic, top_k=config.retrieval.top_k)

    if not existing_chunks:
        # Nothing in the store yet — just add the new info directly
        add(new_info)
        return new_info

    # Summarise existing knowledge into a single block for the merge prompt
    existing_text = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in existing_chunks
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\n"
                f"Existing knowledge:\n{existing_text}\n\n"
                f"New information:\n{new_info}\n\n"
                "Produce the merged note."
            ),
        }
    ]

    merged = complete(messages, system=_MERGE_SYSTEM)

    # Identify which source entry contributed the most retrieved chunks —
    # that's the one to replace (others may be from unrelated sources that
    # just happened to be nearby in embedding space).
    source_counts = Counter(c["source"] for c in existing_chunks)
    dominant_source = source_counts.most_common(1)[0][0]

    # Delete that entry (cascade removes chunks + embeddings)
    conn = init_db()
    try:
        conn.execute("DELETE FROM entries WHERE source = ?", (dominant_source,))
        conn.commit()
    finally:
        conn.close()

    # Re-ingest the merged note as a plain text entry
    add(merged)

    return merged
