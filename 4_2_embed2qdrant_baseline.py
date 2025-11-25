# import_baseline_to_qdrant.py
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

# === 跟你之前保持一致的路径 ===
VS_DIR = Path("baseline_from_indexes_vs")
ENTRIES_PATH = VS_DIR / "baseline_from_idx_entries.json"
EMB_PATH = VS_DIR / "baseline_from_idx_embeddings.npy"

# === Qdrant 连接配置 ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# 和 baseline_chat.py 里用的名字保持一致
BASELINE_COLLECTION = os.getenv("QDRANT_BASELINE_COLLECTION", "baseline_vs")


def main():
    # 1) 读本地 JSON + NPY
    if not ENTRIES_PATH.exists() or not EMB_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find baseline files:\n  {ENTRIES_PATH}\n  {EMB_PATH}\n"
            "Please run build_baseline_from_indexes.py first."
        )

    print(f"Loading entries from {ENTRIES_PATH} ...")
    entries = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))

    print(f"Loading embeddings from {EMB_PATH} ...")
    emb_matrix = np.load(EMB_PATH)
    if emb_matrix.dtype != np.float32:
        emb_matrix = emb_matrix.astype("float32")

    if len(entries) != emb_matrix.shape[0]:
        raise ValueError(
            f"Entries count ({len(entries)}) != embedding rows ({emb_matrix.shape[0]}). "
            "Something is inconsistent."
        )

    dim = emb_matrix.shape[1]
    print(f"Loaded {len(entries)} entries, embedding dim = {dim}")

    # 2) 连接 Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")

    # 3) 创建 / 重建 collection
    print(f"Recreating collection '{BASELINE_COLLECTION}' ...")
    client.recreate_collection(
        collection_name=BASELINE_COLLECTION,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
    )

    # 4) 批量写入 points
    points = []
    for idx, (entry, vec) in enumerate(zip(entries, emb_matrix)):
        # Qdrant 的 id 可以是 int 或 str，这里用 int 最简单
        points.append(
            models.PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload=entry,   # 整个 entry 当作 payload，字段名和 baseline_chat 里用的保持一致
            )
        )

    print(f"Uploading {len(points)} points to '{BASELINE_COLLECTION}' ...")
    # 可以加 batch_size 防止一次太大
    client.upload_points(
        collection_name=BASELINE_COLLECTION,
        points=points,
        batch_size=256,
    )

    print("✅ Import finished.")


if __name__ == "__main__":
    main()
