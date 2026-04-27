import torch
from pythia import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_embedding_model():
    return SentenceTransformer(
        settings.embedding.model,
        trust_remote_code=True,
        device=_DEVICE,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "retrieval"},
    )


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


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    top_k: int | None = None,
) -> list[dict]:
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    hits = client.query_points(
        collection_name=settings.qdrant.collection,
        query=q_vec,
        limit=top_k if top_k is not None else settings.retrieve.top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        results.append(
            {
                "score": hit.score,
                "text": hit.payload["text"],
                "component": hit.payload["component"],
                "page": hit.payload["component"],
                "section_number": hit.payload.get("section_number", "<unknown>")
            }
        )
    return results
