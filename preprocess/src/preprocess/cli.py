import argparse
import sys
from pathlib import Path

from preprocess.index import QDRANT_PATH, chunk_markdown, embed_texts, get_qdrant, upsert_chunks
from preprocess.parse import build_converter, extract_markdown


def cmd_parse(args: argparse.Namespace) -> None:
    converter = build_converter()
    md_text = extract_markdown(converter, args.pdf)

    if args.output:
        Path(args.output).write_text(md_text, encoding="utf-8")
        print(f"Wrote markdown to {args.output}")
    else:
        sys.stdout.write(md_text)


def cmd_index(args: argparse.Namespace) -> None:
    md_text = Path(args.markdown).read_text(encoding="utf-8")
    _index(md_text, component=args.component, qdrant_path=args.qdrant_path)


def cmd_run(args: argparse.Namespace) -> None:
    print(f"Parsing {args.pdf} ...")
    converter = build_converter()
    md_text = extract_markdown(converter, args.pdf)
    _index(md_text, component=args.component, qdrant_path=args.qdrant_path)


def _index(md_text: str, *, component: str, qdrant_path: str) -> None:
    print("Chunking markdown ...")
    chunks = chunk_markdown(md_text, component)
    if not chunks:
        print("No chunks produced — nothing to index.")
        return

    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = embed_texts([c["text"] for c in chunks])

    print(f"Upserting into Qdrant at {qdrant_path!r} ...")
    client = get_qdrant(qdrant_path)
    upsert_chunks(client, chunks, embeddings)
    print(f"Done — indexed {len(chunks)} chunks for component '{component}'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-preprocess",
        description="Preprocessing pipeline: parse PDFs and index them into Qdrant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- parse ---
    p_parse = sub.add_parser("parse", help="Convert a PDF to markdown.")
    p_parse.add_argument("pdf", help="Path to the input PDF file.")
    p_parse.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write markdown to FILE instead of stdout.",
    )
    p_parse.set_defaults(func=cmd_parse)

    # --- index ---
    p_index = sub.add_parser("index", help="Chunk, embed, and upsert a markdown file into Qdrant.")
    p_index.add_argument("markdown", help="Path to the input markdown file.")
    p_index.add_argument("component", help="Component name used as metadata (e.g. 'TPS54331').")
    p_index.add_argument(
        "--qdrant-path",
        default=QDRANT_PATH,
        metavar="PATH",
        help=f"Path to the local Qdrant store (default: {QDRANT_PATH!r}).",
    )
    p_index.set_defaults(func=cmd_index)

    # --- run (full pipeline) ---
    p_run = sub.add_parser("run", help="Parse a PDF and index it in one step.")
    p_run.add_argument("pdf", help="Path to the input PDF file.")
    p_run.add_argument("component", help="Component name used as metadata (e.g. 'TPS54331').")
    p_run.add_argument(
        "--qdrant-path",
        default=QDRANT_PATH,
        metavar="PATH",
        help=f"Path to the local Qdrant store (default: {QDRANT_PATH!r}).",
    )
    p_run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
