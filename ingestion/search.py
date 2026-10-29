from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "./qdrant_data"
COLLECTION = "maintidoc_text"
MODEL_NAME = "intfloat/multilingual-e5-small"

_model = None
_client = None


def _get():
    global _model, _client
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _client = QdrantClient(path=QDRANT_PATH)
    return _model, _client


def search(query: str, user_groups: list[str], top_k: int = 5):
    """Semantic search, filtered BEFORE retrieval by the user's access groups."""
    model, client = _get()
    vec = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    flt = Filter(must=[FieldCondition(key="access_groups", match=MatchAny(any=user_groups))])
    res = client.query_points(
        COLLECTION, query=vec, query_filter=flt, limit=top_k, with_payload=True
    )
    return res.points


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Que faire en cas de surchauffe ?"
    for group in (["tech_pompes"], ["tech_compresseurs"], ["managers"]):
        print(f"\n=== groups={group} | {question}")
        for p in search(question, group):
            d = p.payload
            print(f"[{p.score:.3f}] {d['document_id']} p.{d['page']} ({d['element_id']}, {len(d['text'])} chars)")
            print("   ", d["text"][:250].replace("\n", " "), "...")
    _get()[1].close()