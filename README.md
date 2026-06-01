# research

A lightweight personal research system. Add sources, ask questions,
update memory. Inspired by Andrej Karpathy's LLM Wiki.

No frameworks. Raw API calls. SQLite. Local embeddings.

## Architecture

ingest → chunk → embed → store (SQLite)
query  → embed → search → retrieve → synthesise → answer
update → retrieve existing → merge → re-embed → store

## Setup

1. Clone and install

   git clone git@github.com:krishjoshi/research.git
   cd research
   uv sync

2. Add your API key to .env

   OPENROUTER_API_KEY=your_key_here

## Usage

   # Add a source
   uv run main.py add path/to/nasa_standard.pdf
   uv run main.py add https://nasa.gov/some-doc
   uv run main.py add "Rule 1: no recursion allowed"

   # Ask a question
   uv run main.py ask "what does NASA say about recursion?"

   # Update memory with new information
   uv run main.py update "recursion" "new finding from JPL 2024 standard"

## Stack

- LLM: OpenRouter (Gemini Flash)
- Embeddings: fastembed (local, BAAI/bge-small-en-v1.5)
- Storage: SQLite
- Parsing: pypdf, httpx, beautifulsoup4
