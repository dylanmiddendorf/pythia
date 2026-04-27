import hashlib
import re

import torch
from pythia import settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document as LlamaDocument
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

_NUMBERED_HEADER = re.compile(r"^## <!-- page: (\d+) --> (\d+(?:\.\d)*) (.*)$", re.IGNORECASE | re.MULTILINE)
_FALLBACK_HEADER = re.compile(r"^## <!-- page: (\d+) --> (.*)$", re.IGNORECASE | re.MULTILINE)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _chunk_id(chunk_text: str, component: str) -> int:
    chunk_hash = hashlib.sha256(f"{component}\n\n{chunk_text[:256]}".encode())
    return int.from_bytes(chunk_hash.digest()) % 2**63


def chunk_markdown(md_text: str, component: str) -> list[dict]:
    # Wrap raw markdown in a LlamaIndex Document so the parser can consume it.
    doc = LlamaDocument(text=md_text, metadata={"component": component})

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([doc])

    chunks = []
    for node in nodes:
        chunk_text = node.get_content()
        section_header_text, *section_text = chunk_text.splitlines()

        if (section_header := _NUMBERED_HEADER.match(section_header_text)) is not None:
            page = section_header.group(1)
            section_number = section_header.group(2)
            section_name = section_header.group(3)
            chunk_text = f"## {section_number} {section_name}\n{"\n".join(section_text)}"
        elif (section_header := _FALLBACK_HEADER.match(section_header_text)) is not None:
            page = section_header.group(1)
            section_number = None  # Can't find section number (e.g., "Table of Contents")
            section_name = section_header.group(2)
            chunk_text = f"## {section_name}\n{"\n".join(section_text)}"
        else:
            print(f"[WARNING] Skipping chunk")
            continue

        
        chunks.append(
            {
                "text": chunk_text,
                # Stable ID lets us upsert without duplicates on re-index.
                "chunk_id": _chunk_id(chunk_text, component),
                "metadata": {
                    **node.metadata,
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

    return model.encode(texts, normalize_embeddings=True, batch_size=8, show_progress_bar=True).tolist()


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
