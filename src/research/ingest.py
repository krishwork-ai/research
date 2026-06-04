"""
ingest.py — parse → chunk → embed → store pipeline.

Public API:

    parse_pdf(path)        -> str   extract prose text from a PDF file
    parse_url(url)         -> str   fetch a web page and extract main text
    parse_text(text)       -> str   normalise a raw string (passthrough + clean)
    chunk(text)            -> list[str]   split into overlapping word-windows
    add(source)            -> None  full pipeline: detect source type, parse,
                                    chunk, embed, store

Source-type detection in add():
    ends with .pdf                    → parse_pdf
    starts with http:// or https://   → parse_url
    anything else                     → parse_text

Chunking uses words as the unit (config.chunking.size / overlap).
~512 words ≈ ~680 tokens for English technical prose — close enough to
the config intent without adding a tokeniser dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from research.config import config
from research.store import (
    get_connection,
    init_db,
    insert_entry,
    insert_chunks,
    insert_embeddings,
)
from research.embeddings import embed


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_pdf(path: str | Path) -> str:
    """
    Extract prose text from a PDF using pypdf.

    Strategy: extract text page-by-page, then post-process together.
    We use the simple extract_text() rather than the visitor/y-coord
    approach because NASA docs have varied page geometries and the
    regex cleanup below handles the common noise patterns reliably.

    Post-processing:
      - collapse hyphenated line-breaks ("stan-\\ndard" → "standard")
      - collapse mid-word newlines that pypdf sometimes inserts
      - strip lines that are pure page numbers or very short artefacts
      - normalise whitespace
    """
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    raw = "\n".join(pages)
    return _clean_pdf_text(raw)


def _clean_pdf_text(text: str) -> str:
    # Rejoin hyphenated line-breaks: "stan-\ndard" → "standard"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse mid-word newlines pypdf sometimes inserts inside a word
    # e.g. "recur\nsion" → "recursion"
    # Only lowercase→lowercase: uppercase after a newline is a new sentence/heading.
    text = re.sub(r"(?<=[a-z])\n(?=[a-z])", "", text)

    # Drop lines that are likely page numbers or headers/footers:
    # lines that contain only digits, or are ≤ 3 non-whitespace chars
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # pure page number line
        if re.fullmatch(r"\d+", stripped):
            continue
        # very short artefact (e.g. "1." "A." running headers)
        if len(stripped) <= 3:
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)

    # Normalise runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_url(url: str) -> str:
    """
    Fetch a web page and extract the main readable text.

    Strategy:
      1. httpx GET with a browser-like User-Agent (some NASA pages 403 bots)
      2. BeautifulSoup: remove script/style/nav/header/footer noise
      3. Extract text from the remaining body, normalise whitespace
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; research-bot/1.0; "
            "+https://github.com/krishjoshi/research)"
        )
    }

    try:
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Network error fetching {url}: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise elements
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "noscript",
            "iframe",
        ]
    ):
        tag.decompose()

    # Prefer <main> or <article> if present — better signal for docs sites
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = main.get_text(separator="\n")

    return _clean_generic_text(text)


def parse_text(text: str) -> str:
    """
    Accept a raw string (a user note or annotation).
    Just normalises whitespace — no structural changes.
    """
    return _clean_generic_text(text)


def _clean_generic_text(text: str) -> str:
    # Collapse horizontal whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Drop blank-ish lines (only punctuation/symbols)
    lines = [l for l in lines if re.search(r"[a-zA-Z0-9]", l) or l == ""]
    text = "\n".join(lines)
    # Normalise runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk(text: str) -> list[str]:
    """
    Split text into overlapping word-windows.

    Uses config.chunking.size (words per chunk) and
    config.chunking.overlap (words of overlap between adjacent chunks).

    ~512 words ≈ ~680 tokens for English technical prose — close enough
    to the config intent without a tokeniser dependency.

    Empty or whitespace-only input returns an empty list.
    A text shorter than one full chunk is returned as a single chunk.
    """
    size = config.chunking.size
    overlap = config.chunking.overlap

    if overlap >= size:
        raise ValueError(
            f"chunking.overlap ({overlap}) must be less than chunking.size ({size})"
        )

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Add (full pipeline)
# ---------------------------------------------------------------------------


def _detect_source_type(source: str) -> str:
    s = source.strip()
    if s.lower().endswith(".pdf") or (
        Path(s).exists() and Path(s).suffix.lower() == ".pdf"
    ):
        return "pdf"
    if s.startswith("http://") or s.startswith("https://"):
        return "url"
    return "text"


def add(source: str) -> None:
    """
    Full ingest pipeline: detect source type → parse → chunk → embed → store.

    Parameters
    ----------
    source:
        One of:
          - a file path ending in .pdf
          - a URL starting with http:// or https://
          - any other string (treated as a raw text note)

    Raises
    ------
    ValueError
        If the source is a PDF path that doesn't exist, or if a URL
        returns a non-200 response.
    EmbeddingError
        If the OpenRouter embeddings call fails.
    """
    source_type = _detect_source_type(source)

    # Parse
    if source_type == "pdf":
        path = Path(source)
        if not path.exists():
            raise ValueError(f"PDF not found: {source}")
        text = parse_pdf(path)
    elif source_type == "url":
        text = parse_url(source)
    else:
        text = parse_text(source)

    if not text.strip():
        raise ValueError(f"No text could be extracted from: {source!r}")

    # Chunk
    chunks = chunk(text)
    if not chunks:
        raise ValueError(f"Chunking produced no output for: {source!r}")

    # Embed
    vectors = embed(chunks)

    # Store
    conn = init_db()
    entry_id = insert_entry(conn, source, source_type)
    chunk_ids = insert_chunks(conn, entry_id, chunks)
    insert_embeddings(conn, chunk_ids, vectors, model=config.embeddings.model)
    conn.close()
