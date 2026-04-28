# Pythia

A RAG system for querying electronic component datasheets in natural language. Ask questions like *"What is the input voltage range of the TPS54331?"* and get precise, cited answers grounded in the actual datasheet content.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with `qwen3.5:4b` pulled

## Quickstart

### 1. Install dependencies

```sh
uv sync
```

### 2. Ingest a datasheet

Parse a PDF, chunk by section, embed with Jina, and upsert into the local Qdrant store — all in one step. The component name is stored as metadata on every chunk and surfaced in citations.

```sh
uv run pythia ingest data/raw/TPS54331.pdf TPS54331
```

### 3. Ask a question

```sh
uv run pythia ask "What is the input voltage range of the TPS54331?"
```

**Options:**

| Flag             | Description                                         |
| ---------------- | --------------------------------------------------- |
| `--top-k K`      | Number of chunks to retrieve (default: 5)           |
| `--show-context` | Print retrieved chunks before the answer            |
| `--no-think`     | Disable Qwen extended thinking for faster responses |

## Indexing Multiple Datasheets

Run `ingest` once per datasheet, under its own component name. Queries will then search across all of them and cite the source component.

```sh
uv run pythia ingest data/raw/TPS54331.pdf TPS54331
uv run pythia ingest data/raw/LM2596.pdf LM2596
uv run pythia ingest data/raw/LM5164.pdf LM5164
```

## Retrieval Debugging

The `search` subcommand inspects the vector store independently of the LLM.

```sh
# Run a similarity search and inspect the top-K chunks
uv run pythia search "switching frequency"

# Show the full chunk text instead of the 500-character preview
uv run pythia search "overcurrent protection threshold" --full
```

## Advanced: parse and index separately

`ingest` is the recommended path. If you need to inspect the parsed markdown or re-index without re-parsing, the `parse` and `index` subcommands run each half on its own.

```sh
# PDF → markdown only (writes to stdout, or -o FILE)
uv run pythia parse data/raw/TPS54331.pdf -o data/parsed/TPS54331.md

# markdown → Qdrant only
uv run pythia index data/parsed/TPS54331.md TPS54331
```

## License

BSD 2-Clause. See [LICENSE.txt](LICENSE.txt).
