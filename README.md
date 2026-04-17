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

### 2. Parse a datasheet PDF

Convert a raw PDF into structured markdown. Page numbers are embedded as HTML comments to preserve provenance through chunking.

```sh
uv run preprocess parse data/raw/TPS54331.pdf -o data/parsed/TPS54331.md
```

### 3. Index the parsed markdown

Chunk the markdown by section, embed with Jina, and upsert into the local Qdrant store.

```sh
uv run preprocess index data/parsed/TPS54331.md TPS54331
```

Or do both steps in one command:

```sh
uv run preprocess run data/raw/TPS54331.pdf TPS54331
```

### 4. Ask a question

```sh
uv run assistant "What is the input voltage range of the TPS54331?"
```

**Options:**

| Flag             | Description                                         |
| ---------------- | --------------------------------------------------- |
| `--top-k K`      | Number of chunks to retrieve (default: 5)           |
| `--show-context` | Print retrieved chunks before the answer            |
| `--no-think`     | Disable Qwen extended thinking for faster responses |

## Retrieval Debugging

The `retrieve` CLI lets you inspect and test the vector store independently of the LLM.

```sh
# Show collection statistics (point count, dimensions, distance metric)
uv run retrieve info

# Run a similarity search and inspect the top-K chunks
uv run retrieve search "switching frequency"

# Show full chunk text instead of the 500-character preview
uv run retrieve search "overcurrent protection threshold" --full
```

## Indexing Multiple Datasheets

Index each datasheet under its own component name. The component name is stored as metadata on every chunk and surfaced in citations.

```sh
uv run preprocess run data/raw/TPS54331.pdf TPS54331
uv run preprocess run data/raw/LM2596.pdf LM2596
uv run preprocess run data/raw/LM5164.pdf LM5164
```

After indexing multiple parts, queries will search across all of them and cite the source component in the answer.

## License

BSD 2-Clause. See [LICENSE.txt](LICENSE.txt).
