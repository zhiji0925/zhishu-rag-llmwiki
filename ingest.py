"""全量入库：Raw 文档 -> 章节切片 -> Embedding 向量化 -> LanceDB（含 BM25 全文索引）

用法:
  python ingest.py            # 全量入库
  python ingest.py --limit 3  # 只处理前 3 篇（冒烟测试）
  python ingest.py --stats    # 只看切片统计，不调用 API
"""
import argparse
import time
from collections import Counter

import jieba
import lancedb
import pyarrow as pa

from chunking import chunk_doc, parse_filename
from config import SETTINGS
from embedder import get_embedder
from wiki_graph import extract_entities

TABLE = "chunks"


def tokenize(text: str) -> str:
    return " ".join(jieba.cut_for_search(text))


def build_rows(limit=None):
    files = sorted(SETTINGS.raw_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"未找到文档目录: {SETTINGS.raw_dir}")
    if limit:
        files = files[:limit]
    rows = []
    for f in files:
        customer, project_type, doc_type, doc_date = parse_filename(f.stem)
        title, chunks = chunk_doc(f.read_text(encoding="utf-8"))
        for ch in chunks:
            if len(ch["text"]) < 30 and ch["section"] not in ("文档信息", "文档元信息"):
                continue  # 过滤无内容碎片
            rows.append({
                "doc_name": f.stem,
                "customer": customer,
                "project_type": project_type,
                "doc_type": doc_type,
                "doc_date": doc_date,
                "title": title,
                "section": ch["section"],
                "subsection": ch["subsection"],
                "text": ch["text"],
                "text_tokens": tokenize(f"{ch['section']} {ch['subsection']} {ch['text']}"),
                "entities": " ".join(extract_entities(ch["text"])),
            })
    print(f"文档 {len(files)} 篇 -> 切片 {len(rows)} 条（平均 {len(rows) / max(len(files), 1):.1f} 片/篇）")
    return rows


def to_arrow(rows, vectors):
    dim = len(vectors[0])
    schema = pa.schema([
        ("id", pa.int64()),
        ("vector", pa.list_(pa.float32(), dim)),
        ("text", pa.string()),
        ("text_tokens", pa.string()),
        ("entities", pa.string()),
        ("doc_name", pa.string()),
        ("customer", pa.string()),
        ("project_type", pa.string()),
        ("doc_type", pa.string()),
        ("doc_date", pa.string()),
        ("section", pa.string()),
        ("subsection", pa.string()),
        ("title", pa.string()),
    ])
    data = [{**r, "id": i, "vector": v} for i, (r, v) in enumerate(zip(rows, vectors))]
    return pa.Table.from_pylist(data, schema=schema)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows = build_rows(args.limit)

    if args.stats:
        print("按文档类型分布:", dict(Counter(r["doc_type"] for r in rows)))
        lens = sorted(len(r["text"]) for r in rows)
        print(f"切片字符数: min={lens[0]} p50={lens[len(lens) // 2]} max={lens[-1]}")
        return

    print(f"Embedding: provider={SETTINGS.embedding_provider} model={SETTINGS.embedding_model} dim={SETTINGS.embedding_dim}")
    vectors = get_embedder().embed_documents([r["text"] for r in rows])

    db = lancedb.connect(str(SETTINGS.lancedb_dir))
    table = db.create_table(TABLE, to_arrow(rows, vectors), mode="overwrite")
    table.create_fts_index("text_tokens")
    print(f"完成: {len(rows)} 切片已写入 {SETTINGS.lancedb_dir}（含 BM25 索引），耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
