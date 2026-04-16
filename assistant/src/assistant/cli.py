from __future__ import annotations

import argparse
import re
import textwrap
import torch
import gc

from ollama import chat, ChatResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"
EMBEDDING_DIM = 1024
QDRANT_PATH = "data/qdrant_store"
QDRANT_COLLECTION = "datasheets"
OLLAMA_MODEL = "qwen3.5:4b"
TOP_K = 5

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a technical assistant that answers questions about electronic
    component datasheets. You will be given retrieved context passages from
    one or more DC-DC converter datasheets.

    Rules:
    1. Answer ONLY from the provided context. If the context does not contain
       enough information, say so explicitly — do not guess or hallucinate.
    2. When citing a specific value (voltage, current, frequency, etc.),
       include the unit and any qualifying conditions (e.g. temperature, input
       voltage, load current).
    3. If the answer involves a range (min/typ/max), state all available
       bounds.
    4. When the context contains a table or enumerated list of thresholds
       relevant to the question, include all rows/entries rather than
       summarizing with a single example.
    5. If a parameter behaves differently under specific conditions (e.g.,
       fault, light load, startup), explain what triggers the change and
       how the behavior differs from normal operation.
    6. If the context contains apparently contradictory specifications, note
       and reconcile them (e.g., a "fixed" frequency that changes during fault
       conditions).
    7. Cite the specific sections that support your answer using
       [Section x.y.z, Page n] format. If no section number is available, cite
       the page number.
    8. Keep answers focused and avoid restating the question, but prioritize
       completeness of technical detail over brevity.
"""
)


def build_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    context_block = ""
    for chunk in context_chunks:
        source = chunk.get("component", "unknown")
        text = chunk["text"]
        if chunk["section_number"] is not None:
            context_block += f"[Page {chunk["page"]}, Section {chunk["section_number"]} — {source}]\n{text}\n\n"
        else:
            context_block += f"[Page {chunk["page"]} — {source}]\n{text}\n\n"

    user_content = (
        f"Context:\n{context_block}" f"Question: {query}\n\n" f"Answer the question using only the context above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    top_k: int = TOP_K,
) -> list[dict]:
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        results.append(
            {
                "score": hit.score,
                "text": hit.payload["text"],
                "component": hit.payload["component"],
                "page": hit.payload["page"],
                "section_number": hit.payload.get("section_number", "<unknown>"),
            }
        )
    return results

def generate(
    query: str,
    context_chunks: list[dict],
    think: bool = True,
) -> str:
    messages = build_prompt(query, context_chunks)

    response: ChatResponse = chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0.2,  # low temp for factual grounding
            "num_ctx": 8192,  # context window for long chunks
            "num_predict": 4096,  # cap output length
        },
        think=think,
    )

    content = response.message.content or ""
    thinking = response.message.thinking or ""

    return content, thinking


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks if present in raw output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Datasheet — RAG generation PoC")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Qwen thinking mode for faster responses",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks before the answer",
    )
    args = parser.parse_args()

    print(f"[embed] Loading {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16, "default_task": "retrieval"},  # Recommended for GPUs
    )
    client = QdrantClient(path=QDRANT_PATH)

    print(f"[retrieve] Searching for top-{args.top_k} chunks ...")
    chunks = retrieve(args.query, embedder, client, top_k=args.top_k)

    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    if not chunks:
        print("[retrieve] No chunks found. Have you indexed any datasheets?")
        return

    if args.show_context:
        print(f"\n{'=' * 60}")
        print("RETRIEVED CONTEXT")
        print(f"{'=' * 60}")
        for i, c in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} (score: {c['score']:.4f}, src: {c['source']}) ---")
            print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

    print(f"\n[generate] Querying {OLLAMA_MODEL} ...")
    answer, thinking = generate(args.query, chunks, think=not args.no_think)

    if thinking:
        print(f"\n{'=' * 60}")
        print("MODEL THINKING (internal reasoning)")
        print(f"{'=' * 60}")
        print(thinking)

    print(f"\n{'=' * 60}")
    print(f"Q: {args.query}")
    print(f"{'=' * 60}")
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
