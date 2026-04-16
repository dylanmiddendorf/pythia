import argparse

from retrieve import (
    QDRANT_COLLECTION,
    QDRANT_PATH,
    TOP_K,
    get_embedding_model,
    get_qdrant,
    retrieve,
)


def cmd_search(args: argparse.Namespace) -> None:
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


def cmd_info(args: argparse.Namespace) -> None:
    client = get_qdrant(args.qdrant_path)

    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        print(f"Collection '{QDRANT_COLLECTION}' does not exist yet.")
        return

    info = client.get_collection(QDRANT_COLLECTION)
    count = client.count(QDRANT_COLLECTION).count
    cfg = info.config.params.vectors

    print(f"Collection : {QDRANT_COLLECTION}")
    print(f"Points     : {count:,}")
    print(f"Dimensions : {cfg.size}")
    print(f"Distance   : {cfg.distance.value}")
    print(f"Qdrant path: {args.qdrant_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-retrieve",
        description="Retrieval testing CLI — query the Qdrant datasheet index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- search ---
    p_search = sub.add_parser("search", help="Run a retrieval query and print results.")
    p_search.add_argument("query", help="Natural language question.")
    p_search.add_argument(
        "--top-k", type=int, default=TOP_K, metavar="K",
        help=f"Number of results to return (default: {TOP_K}).",
    )
    p_search.add_argument(
        "--qdrant-path", default=QDRANT_PATH, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {QDRANT_PATH!r}).",
    )
    p_search.add_argument(
        "--full", action="store_true",
        help="Print the full chunk text instead of a 500-character preview.",
    )
    p_search.set_defaults(func=cmd_search)

    # --- info ---
    p_info = sub.add_parser("info", help="Show collection stats.")
    p_info.add_argument(
        "--qdrant-path", default=QDRANT_PATH, metavar="PATH",
        help=f"Path to the local Qdrant store (default: {QDRANT_PATH!r}).",
    )
    p_info.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
