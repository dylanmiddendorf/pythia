from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from pythia import settings


def cmd_parse(args: argparse.Namespace) -> None:
    from pythia.parse import build_converter, extract_markdown

    converter = build_converter()
    md_text = extract_markdown(converter, args.pdf)

    if args.output:
        Path(args.output).write_text(md_text, encoding="utf-8")
        print(f"Wrote markdown to {args.output}")
    else:
        sys.stdout.write(md_text)


def cmd_ingest(args: argparse.Namespace) -> None:
    from pythia.ingest import chunk_markdown, embed_texts, get_qdrant, upsert_chunks

    md_text = Path(args.markdown).read_text(encoding="utf-8")
    print("Chunking markdown ...")
    chunks = chunk_markdown(md_text, args.component)
    if not chunks:
        print("No chunks produced — nothing to index.")
        return

    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = embed_texts([c["text"] for c in chunks])

    print(f"Upserting into Qdrant at {args.qdrant_path!r} ...")
    client = get_qdrant(args.qdrant_path)
    upsert_chunks(client, chunks, embeddings)
    print(f"Done — indexed {len(chunks)} chunks for component '{args.component}'.")


def cmd_add(args: argparse.Namespace) -> None:
    from pythia.parse import build_converter, extract_markdown
    from pythia.ingest import chunk_markdown, embed_texts, get_qdrant, upsert_chunks

    print(f"Parsing {args.pdf} ...")
    converter = build_converter()
    md_text = extract_markdown(converter, args.pdf)

    print("Chunking markdown ...")
    chunks = chunk_markdown(md_text, args.component)
    if not chunks:
        print("No chunks produced — nothing to index.")
        return

    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = embed_texts([c["text"] for c in chunks])

    print(f"Upserting into Qdrant at {args.qdrant_path!r} ...")
    client = get_qdrant(args.qdrant_path)
    upsert_chunks(client, chunks, embeddings)
    print(f"Done — indexed {len(chunks)} chunks for component '{args.component}'.")


def cmd_search(args: argparse.Namespace) -> None:
    from pythia.retrieve import get_embedding_model, get_qdrant, retrieve

    print("Loading embedding model ...")
    model = get_embedding_model()
    client = get_qdrant(args.qdrant_path)

    print(f"Searching for top-{args.top_k} results ...\n")
    results = retrieve(args.query, model, client, top_k=args.top_k)

    if not results:
        print("No results found. Has anything been indexed yet?")
        return

    print(f"Query : {args.query}")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        section = r["section_number"] or "—"
        print(f"\n[{i}] score={r['score']:.4f}  component={r['component']}  page={r['page']}  section={section}")
        text = r["text"]
        if not args.full and len(text) > 500:
            text = text[:500] + "..."
        print(text)
    print()


def cmd_ask(args: argparse.Namespace) -> None:
    import torch

    from pythia.generate import generate
    from pythia.retrieve import get_embedding_model, get_qdrant, retrieve

    print(f"[embed] Loading {settings.embedding.model} ...")
    embedder = get_embedding_model()
    client = get_qdrant(args.qdrant_path)

    print(f"[retrieve] Searching for top-{args.top_k} chunks ...")
    chunks = retrieve(args.query, embedder, client, top_k=args.top_k)

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not chunks:
        print("[retrieve] No chunks found. Have you indexed any datasheets?")
        return

    if args.show_context:
        print(f"\n{'=' * 60}\nRETRIEVED CONTEXT\n{'=' * 60}")
        for i, c in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} (score: {c['score']:.4f}, src: {c['component']}) ---")
            print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

    print(f"\n[generate] Querying {settings.generation.ollama_model} ...")
    answer, thinking = generate(args.query, chunks, think=not args.no_think)

    if thinking:
        print(f"\n{'=' * 60}\nMODEL THINKING (internal reasoning)\n{'=' * 60}")
        print(thinking)

    print(f"\n{'=' * 60}\nQ: {args.query}\n{'=' * 60}\n\n{answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia",
        description="Pythia — RAG over electronic component datasheets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest (parse + index) ---
    p_ingest = sub.add_parser("add", help="Parse a PDF and index it into Qdrant in one step.")
    p_ingest.add_argument("pdf", help="Path to the input PDF file.")
    p_ingest.add_argument("component", help="Component name used as metadata (e.g. 'TPS54331').")
    p_ingest.add_argument(
        "--qdrant-path", default=settings.qdrant.path, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {settings.qdrant.path!r}).",
    )
    p_ingest.set_defaults(func=cmd_add)

    # --- parse (PDF → markdown only) ---
    p_parse = sub.add_parser("parse", help="Convert a PDF to markdown without indexing.")
    p_parse.add_argument("pdf", help="Path to the input PDF file.")
    p_parse.add_argument("-o", "--output", metavar="FILE", help="Write markdown to FILE instead of stdout.")
    p_parse.set_defaults(func=cmd_parse)

    # --- index (markdown → Qdrant only) ---
    p_index = sub.add_parser("ingest", help="Chunk, embed, and upsert a pre-parsed markdown file.")
    p_index.add_argument("markdown", help="Path to the input markdown file.")
    p_index.add_argument("component", help="Component name used as metadata.")
    p_index.add_argument(
        "--qdrant-path", default=settings.qdrant.path, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {settings.qdrant.path!r}).",
    )
    p_index.set_defaults(func=cmd_ingest)

    # --- search (retrieval only) ---
    p_search = sub.add_parser("search", help="Run a retrieval query against the Qdrant index.")
    p_search.add_argument("query", help="Natural language query.")
    p_search.add_argument(
        "--top-k", type=int, default=settings.retrieval.top_k, metavar="K",
        help=f"Number of results to return (default: {settings.retrieval.top_k}).",
    )
    p_search.add_argument(
        "--qdrant-path", default=settings.qdrant.path, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {settings.qdrant.path!r}).",
    )
    p_search.add_argument("--full", action="store_true", help="Print full chunk text (no 500-char preview).")
    p_search.set_defaults(func=cmd_search)

    # --- ask (retrieval + generation) ---
    p_ask = sub.add_parser("ask", help="Ask a natural-language question grounded in the indexed datasheets.")
    p_ask.add_argument("query", help="Natural language question.")
    p_ask.add_argument(
        "--top-k", type=int, default=settings.retrieval.top_k,
        help=f"Number of chunks to retrieve (default: {settings.retrieval.top_k}).",
    )
    p_ask.add_argument(
        "--qdrant-path", default=settings.qdrant.path, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {settings.qdrant.path!r}).",
    )
    p_ask.add_argument("--no-think", action="store_true", help="Disable model thinking for faster responses.")
    p_ask.add_argument("--show-context", action="store_true", help="Print retrieved chunks before the answer.")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
