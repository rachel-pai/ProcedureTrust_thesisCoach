from qdrant_client import QdrantClient, models
import json
from pathlib import Path
import uuid

POLICY_INDEX_PATH = Path("policy_docs_index.json")
THESES_INDEX_PATH = Path("thesis_corpus_index.json")

client = QdrantClient(
    url="https://36d9f3c2-f784-4246-a7e2-f8d24c8be6a3.eu-west-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.fEDS2wkdVl7aMtQqzbLCMnZE9_WEy6qZnLgmGCoEZOI",
)

print(client.get_collections())

# 你 embeddings 维度（text-embedding-3-large 是 3072）
EMBED_DIM = 3072

from qdrant_client import QdrantClient, models

def ensure_collection(name: str):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )

ensure_collection("policy_docs")
ensure_collection("thesis_segments")


def migrate_policy():
    data = json.loads(POLICY_INDEX_PATH.read_text(encoding="utf-8"))
    points = []

    for doc in data.get("docs", []):
        doc_source_path = doc.get("source_path")
        for item in doc.get("items", []):
            emb = item.get("embedding")
            if not emb:
                continue

            payload = {
                "raw_id": item["id"],
                "label": item["label"],
                "description": item["description"],
                "risk_level": item.get("risk_level"),
                "doc_title": item.get("doc_title"),
                "doc_stage": item.get("doc_stage"),
                "doc_mode": item.get("doc_mode"),
                "item_stage": item.get("item_stage", item.get("doc_stage")),
                "item_mode": item.get("item_mode", item.get("doc_mode")),
                "source_path": item.get("source_path", doc_source_path),
                "source_chunk_md": item.get("source_chunk_md"),
            }

            points.append(
                models.PointStruct(
                    # id=item["id"],
                    id=str(uuid.uuid4()),  # Qdrant 要求的 point ID
                    vector=emb,
                    payload=payload,
                )
            )

    print("total policy points:", len(points))

    # ✅ 分批 upsert，避免单次请求超过 32MB
    BATCH = 200      # 200–500 都可以
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        client.upsert(
            collection_name="policy_docs",
            points=batch,
        )
        print(f"upserted policy batch {i}–{i+len(batch)-1}")

    print("Migrated all policy items.")



def migrate_theses():
    text = THESES_INDEX_PATH.read_text(encoding="utf-8")
    data = json.loads(text)

    blocks = []
    if isinstance(data, dict) and "docs" in data:
        blocks = [data]
    elif isinstance(data, list):
        flat = []
        for x in data:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        blocks = flat

    points = []

    for block in blocks:
        for doc in block.get("docs", []):
            for seg in doc["segments"]:
                emb = seg.get("embedding")
                if not emb:
                    continue
                payload = {
                    "raw_id": seg["id"],
                    "label": seg.get("label", ""),
                    "summary": seg.get("summary") or seg.get("description") or "",
                    "stage": seg.get("stage", seg.get("item_stage", "other")),
                    "mode": seg.get("mode", seg.get("item_mode", "precedents")),
                    "field": seg.get("field", "unknown"),
                    "source_path": seg.get("source_path"),
                    "doc_title": seg.get("doc_title"),
                    "source_type": seg.get("source_type", "thesis"),
                    "role": seg.get("role", "technical_precedent"),
                    "domain_tags": seg.get("domain_tags", []),
                    "construct_tags": seg.get("construct_tags", []),
                    "user_tags": seg.get("user_tags", []),
                    "metric_tags": seg.get("metric_tags", []),
                    "raw_excerpt_md": seg.get("raw_excerpt_md"),
                    "source_chunk_md": seg.get("source_chunk_md"),
                }

                points.append(
                    models.PointStruct(
                        # id=seg["id"],
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload=payload,
                    )
                )

    print("total thesis points:", len(points))

    BATCH = 200
    for i in range(0, len(points), BATCH):
        batch = points[i:i + BATCH]
        client.upsert(
            collection_name="thesis_segments",
            points=batch,
        )
        print(f"upserted thesis batch {i}–{i + len(batch) - 1}")

    print("Migrated all thesis segments.")

    print(f"Migrated {len(points)} thesis segments")


if __name__ == "__main__":
    ensure_collection("policy_docs")
    ensure_collection("thesis_segments")
    migrate_policy()
    migrate_theses()


