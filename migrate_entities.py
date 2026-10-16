"""一次性迁移：为已有 LanceDB 表补 entities 列（复用已存向量，无需重新 embedding）"""
import lancedb

from config import SETTINGS
from ingest import TABLE, to_arrow
from wiki_graph import extract_entities

db = lancedb.connect(str(SETTINGS.lancedb_dir))
rows = db.open_table(TABLE).to_arrow().to_pylist()
hit = 0
for r in rows:
    r["entities"] = " ".join(extract_entities(r["text"]))
    hit += bool(r["entities"])
vectors = [r.pop("vector") for r in rows]
table = db.create_table(TABLE, to_arrow(rows, vectors), mode="overwrite")
table.create_fts_index("text_tokens")
print(f"迁移完成: {len(rows)} 行，其中 {hit} 行含实体")
