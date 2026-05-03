import hashlib
import re

import torch
from pythia import settings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Major-section headers carry an injected page comment plus an optional section
# number (e.g. "<!-- page: 4 --> 6 Specifications" or "<!-- page: 2 --> Revision History").
_HEADER_COMMENT = re.compile(r"<!-- page: (\d+) --> (?:(\d+(?:\.\d+)*) )?(.*)")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _chunk_id(chunk_text: str, component: str) -> int:
    chunk_hash = hashlib.sha256(f"{component}\n\n{chunk_text[:256]}".encode())
    return int.from_bytes(chunk_hash.digest()) % 2**63


def chunk_markdown(md_text: str, component: str) -> list[dict]:
    # Split only on first- and second-level headers; sub-sections stay inside
    # the parent chunk so each major section is retrieved as a coherent unit.
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )
    docs = splitter.split_text(md_text)

    chunks = []
    for doc in docs:
        # Skip the document-title chunk; only level-2 sections carry indexable content.
        raw_header = doc.metadata.get("h2")
        if raw_header is None:
            continue

        header_match = _HEADER_COMMENT.match(raw_header)
        if header_match is None:
            print(f"[WARNING] Skipping chunk with unrecognized header: {raw_header!r}")
            continue
        
        # Drop the page-tracking artifacts
        page, section_number, section_name = header_match.groups()
        clean_header = f"## {section_number} {section_name}" if section_number else f"## {section_name}"
        _, *body = doc.page_content.splitlines()
        chunk_text = "\n".join([clean_header, *body])

        chunks.append(
            {
                "text": chunk_text,
                # Stable ID lets us upsert without duplicates on re-index.
                "chunk_id": _chunk_id(chunk_text, component),
                "metadata": {
                    "component": component,
                    "page": page,
                    "section_number": section_number,
                },
            }
        )

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = SentenceTransformer(
        settings.embedding.model,
        trust_remote_code=True,
        device=_DEVICE,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "retrieval"},
    )

    return model.encode(texts, normalize_embeddings=True, batch_size=2, show_progress_bar=True).tolist()


def get_qdrant(path: str | None = None) -> QdrantClient:
    client = QdrantClient(path=path or settings.qdrant.path)

    # Create collection if it doesn't exist yet.
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant.collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant.collection,
            vectors_config=VectorParams(
                size=settings.embedding.dim,
                distance=Distance.COSINE,
            ),
        )

    return client


def upsert_chunks(client: QdrantClient, chunks: list[dict], embeddings: list[list[float]]):
    points = []
    for chunk, vec in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=vec,
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
        )

    client.upsert(collection_name=settings.qdrant.collection, points=points)
