import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

TOP_K = 5
QDRANT_PATH = "data/qdrant_store"
QDRANT_COLLECTION = "datasheets"

EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small"
EMBEDDING_DIM = 1024  # https://huggingface.co/jinaai/jina-embeddings-v5-text-small

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        device=_DEVICE,
        model_kwargs={"dtype": torch.bfloat16, "default_task": "retrieval"},
    )


def get_qdrant(path: str = QDRANT_PATH) -> QdrantClient:
    client = QdrantClient(path=path)

    # Create collection if it doesn't exist yet.
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    return client


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
                "page": hit.payload["component"],
                "section_number": hit.payload.get("section_number", "<unknown>")
            }
        )
    return results
