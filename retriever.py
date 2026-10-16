"""检索模块：向量(cosine) + BM25 全文 + 图谱(实体倒排) 多路召回，RRF 融合

模式:
  auto   - 由路由器判定难度: 简单=向量+BM25(+元数据预过滤); 复杂=再叠加图谱召回;
           实体=按实体名扫描 Raw 文件的 [[双链]] 命中段落(跳过关键词检索与向量检索);
           关键词全是泛概念时不触发实体扫描，退回简单模式
  hybrid - 手动双路(向量+BM25)
  vector / fts - 单路
  entity - 手动实体扫描

返回条数一律按调用方传入的 top_k（复杂模式下具体实体链接到的文档会补全、可超限；
泛概念只参与排序加分、不触发补全，避免几十篇关联文档灌满提示词）；
仅实体扫描模式不设篇数上限：命中文件全部进提示词，但每篇只注入命中段落(<= ENTITY_MAX_CHARS)。
"""
import re
import time
from collections import Counter

import jieba
import lancedb

from chunking import parse_filename
from config import SETTINGS
from embedder import get_embedder
from wiki_graph import get_dictionary

TABLE = "chunks"
RRF_K = 60
ENTITY_MAX_CHARS = 800                 # 实体模式：单篇文件最多注入的字符数（命中段落 + 邻近小节）
ENTITY_MAX_BLOCKS = 4                  # 实体模式：单篇最多保留的命中块数
ENTITY_FULL_MAX = 5                    # 实体模式：命中文档少于此数时，整篇原文进提示词
ENTITY_SOFT_RATIO = 0.4                # 命中文档占比超过该值即视为泛概念(只参与排序)
_LINK_MARK = re.compile(r"\[\[|\]\]")   # Raw 正文的双链标记，展示时剥离
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
_H = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FIELDS = ("id", "text", "doc_name", "customer", "project_type",
          "doc_type", "doc_date", "section", "subsection", "title", "entities")


def _clip(text: str, limit: int) -> str:
    """按行边界截断，避免把表格/句子切一半"""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[:cut if cut > limit * 0.5 else limit].rstrip() + " …"


def _raw_blocks(text: str) -> list[tuple[tuple[str, str], str]]:
    """Raw 原文按标题切块 -> [((h2, h3), 正文原文), ...]，保留 [[ ]] 标记供命中判定"""
    out, h2, h3, buf = [], "", "", []

    def flush():
        nonlocal buf
        body = "\n".join(l for l in buf if l.strip() and l.strip() != "---").strip()
        if body:
            out.append(((h2, h3), body))
        buf = []

    for line in text.splitlines():
        m = _H.match(line)
        if m:
            lvl, name = len(m.group(1)), m.group(2)
            if lvl == 1:
                continue                      # 文档大标题不入块
            flush()
            if lvl == 2:
                h2, h3 = name, ""
            else:
                h3 = name
        else:
            buf.append(line)
    flush()
    return out


def _entity_excerpt(blocks: list[tuple[tuple[str, str], str]], names: list[str]):
    """命中块(最多 ENTITY_MAX_BLOCKS 块) + 前后各一块，拼装成不超过 ENTITY_MAX_CHARS 的摘录
    -> (正文, 章节标题列表)"""
    hits = [i for i, (_, b) in enumerate(blocks) if any(f"[[{n}]]" in b for n in names)]
    if not hits:
        return "", []
    if len(hits) > ENTITY_MAX_BLOCKS:   # 命中块过多时，按块内实体命中处数取前几块
        hits = sorted(sorted(hits, key=lambda i: -sum(blocks[i][1].count(f"[[{n}]]")
                                                      for n in names))[:ENTITY_MAX_BLOCKS])

    budget, picked = ENTITY_MAX_CHARS, {}
    per = max(120, budget // len(hits) - 2)     # 命中块平均分配预算，必留
    for i in hits:
        picked[i] = _clip(blocks[i][1], per)
        budget -= len(picked[i])
    nbr = {j + d for j in hits for d in (-1, 1)} - set(hits)
    for i in sorted(i for i in nbr if 0 <= i < len(blocks)):   # 剩余预算补邻近小节作上下文
        if len(blocks[i][1]) <= budget:
            picked[i] = blocks[i][1]
            budget -= len(blocks[i][1]) + 2

    idx = sorted(picked)
    body = _LINK_MARK.sub("", "\n\n".join(picked[i] for i in idx)).strip()
    titles = [" › ".join(t for t in blocks[i][0] if t) or "文档信息" for i in idx]
    return body, titles


def _hit(row: dict) -> dict:
    item = {k: row.get(k) for k in FIELDS}
    item["scores"] = {}
    if "_distance" in row:  # cosine 距离 -> 相似度
        item["scores"]["vector"] = round(1 - row["_distance"], 4)
    if "_score" in row:     # BM25 得分
        item["scores"]["fts"] = round(row["_score"], 2)
    return item


def _sql(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


_SORT_KEYS = ("rrf", "vector", "fts", "graph", "entity")


def _by_score(items: list[dict]) -> list[dict]:
    """按融合分 RRF 降序；单通道模式没有 RRF 时退化为该通道得分降序。
    用 RRF 而非纯相似度：只有向量通道有相似度，纯相似度排序会把
    仅被 BM25 / 图谱命中的切片全部沉底。"""
    return sorted(items,
                  key=lambda x: tuple(x["scores"].get(c, -1) for c in _SORT_KEYS),
                  reverse=True)


class Retriever:
    def __init__(self):
        self.db = lancedb.connect(str(SETTINGS.lancedb_dir))
        self.table = self.db.open_table(TABLE)
        self.embedder = get_embedder()

    # ---- 通道 ----
    def _vector(self, query: str, k: int, where: str | None = None) -> dict:
        vec = self.embedder.embed_query(query)
        builder = self.table.search(vec).distance_type("cosine")
        if where:
            builder = builder.where(where)
        hits = builder.limit(k * 2).to_list()
        return {h["id"]: _hit(h) for h in hits}  # to_list 已按距离排序

    def _fts(self, query: str, k: int, where: str | None = None) -> dict:
        tokens = " ".join(jieba.cut_for_search(query))
        builder = self.table.search(tokens, query_type="fts", fts_columns=["text_tokens"])
        if where:
            builder = builder.where(where)
        hits = builder.limit(k * 2).to_list()
        return {h["id"]: _hit(h) for h in hits}

    def _graph(self, entities: list[str], k: int, where: str | None = None) -> dict:
        """图谱召回：Wiki 实体 -> Raw 双链倒排(contains)，按命中实体数排序"""
        cond = " OR ".join(f"contains(entities, {_sql(e)})" for e in entities)
        if where:
            cond = f"({cond}) AND ({where})"
        try:
            hits = self.table.search().where(cond).limit(2000).to_list()
        except Exception:
            return {}
        scored = []
        for h in hits:
            ents = h.get("entities") or ""
            item = _hit(h)
            item["scores"]["graph"] = sum(1 for e in entities if e in ents)
            scored.append(item)
        scored.sort(key=lambda x: -x["scores"]["graph"])
        return {item["id"]: item for item in scored}

    # ---- 实体扫描（第三种检索方式：跳过关键词检索与向量检索）----
    def _entity_scan(self, names: list[str]) -> dict:
        """按实体/概念名扫描 Raw 文件，命中文件各出一条结果（不做条数上限）。

        分层召回（避免高频概念把几十篇文档全灌进提示词）：
        - 具体实体(客户/项目类型/人名/部门/技术组件) 作硬约束，多关键词取**交集**；
        - 概念、以及命中文档占比 > ENTITY_SOFT_RATIO 的泛概念，只参与**排序加分**；
        - 交集为空时逐级回退并集，绝不归零；
        - 同类多实体(如 中国移动+中国联通) 视为对比场景，不取交集、各取一批。
        每篇只注入 [[实体]] 命中段落 + 前后各一小节（<= ENTITY_MAX_CHARS）；
        命中文档不足 ENTITY_FULL_MAX 篇时，改为整篇原文进提示词（材料少则给全）。
        -> {"docs": 结果, "note": 召回说明, "hard": 硬约束词, "soft": 泛概念词}
        """
        names = [n for n in names if n]
        per_doc: dict[str, dict[str, int]] = {}   # 文档 -> {关键词: [[ ]] 命中处数}
        texts: dict[str, str] = {}
        total = 0
        for f in sorted(SETTINGS.raw_dir.glob("*.md")):
            total += 1
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            cnt = {n: c for n, c in ((n, text.count(f"[[{n}]]")) for n in names) if c}
            if cnt:
                per_doc[f.stem] = cnt
                texts[f.stem] = text
        if not per_doc:
            return {"docs": [], "note": "无命中文件", "hard": [], "soft": []}

        dic = get_dictionary()
        doc_hits = {n: {s for s, c in per_doc.items() if n in c} for n in names}
        soft = {n for n in names if dic.get(n, "entity") == "concept"
                or len(doc_hits[n]) > total * ENTITY_SOFT_RATIO}
        hard = [n for n in names if n not in soft]
        by_type = Counter(dic.get(n, "entity") for n in hard)
        single = [n for n in hard if by_type[dic.get(n, "entity")] == 1]   # 跨轴约束
        compare = [n for n in hard if by_type[dic.get(n, "entity")] > 1]   # 同类多实体=对比

        def union(ns):
            return set().union(*(doc_hits[n] for n in ns)) if ns else set()

        if single:
            cand = set.intersection(*(doc_hits[n] for n in single))
            mode = f"交集 {len(single)} 词"
        else:
            cand, mode = union(hard), "并集"
        if not cand:                                  # 交集为空 -> 逐级回退，避免归零
            cand, mode = union(hard), "回退并集"
        if not cand:
            cand, mode = union(names), "回退并集(含泛概念)"
        cand |= union(compare)

        # 命中文档少 -> 整篇原文进提示词（材料本来就少，给全比截段落更利于作答）
        full = len(cand) < ENTITY_FULL_MAX
        docs = []
        for stem in cand:
            text = texts[stem]
            customer, project_type, doc_type, doc_date = parse_filename(stem)
            m = _H1.search(text)
            matched = per_doc[stem]
            if full:
                section, body = "全文", _LINK_MARK.sub("", text).strip()
            else:
                excerpt, titles = _entity_excerpt(_raw_blocks(text), names)
                section = titles[0] if titles else "命中段落"
                body = excerpt or _clip(_LINK_MARK.sub("", text).strip(), ENTITY_MAX_CHARS)
            docs.append({
                "doc_name": stem, "customer": customer, "project_type": project_type,
                "doc_type": doc_type, "doc_date": doc_date,
                "title": m.group(1).strip() if m else "",
                "section": section,
                "subsection": "、".join(f"[[{n}]] 命中 {c} 处" for n, c in matched.items()),
                "matched": list(matched),
                "text": body,
                # 排序：硬约束命中处数优先，泛概念命中只作加成分
                "scores": {"entity": sum(matched.values()),
                           "entity_hard": sum(c for n, c in matched.items() if n in hard)},
            })
        docs.sort(key=lambda x: (-x["scores"]["entity_hard"], -x["scores"]["entity"]))

        note = ["无具体实体约束（泛概念并集）"] if not hard else \
               [f"具体实体取{mode}" + (f"（{'、'.join(single)}）" if single else "")]
        if compare:
            note.append(f"对比实体各取一批（{'、'.join(compare)}）")
        if soft:
            note.append(f"泛概念仅排序加分（{'、'.join(sorted(soft))}）")
        return {"docs": docs,
                "note": "；".join(note) + f" → {len(docs)} 篇"
                        + (f"（不足 {ENTITY_FULL_MAX} 篇，注入整篇原文）" if full
                           else f"（每篇注入 [[实体]] 命中段落）"),
                "hard": hard, "soft": sorted(soft), "full": full}

    # ---- 辅助 ----
    @staticmethod
    def _dedup_docs(items: list[dict]) -> list[dict]:
        """同一文档的多个切片只保留融合分最高的那个。
        图谱通道是文档级召回，容易让同一文档的多个切片霸屏。"""
        seen, out = set(), []
        for it in items:  # 入参已按融合分降序
            if it["doc_name"] in seen:
                continue
            seen.add(it["doc_name"])
            out.append(it)
        return out

    @staticmethod
    def _complete_by_entity(items: list[dict], graph_hits: dict, k: int) -> list[dict]:
        """复杂模式：先取融合 top-k，再把图谱(实体双链)链接到的文档全部补全进来。
        实体链接的文档数超过 top_k 时允许突破条数限制，让答案更全。"""
        by_doc = {}
        for it in graph_hits.values():  # 同一文档取实体命中数最多的切片
            d = it["doc_name"]
            if d not in by_doc or it["scores"].get("graph", 0) > by_doc[d]["scores"].get("graph", 0):
                by_doc[d] = it
        head = items[:k]
        seen = {it["doc_name"] for it in head}
        extra = sorted((by_doc[d] for d in by_doc if d not in seen),
                       key=lambda x: -x["scores"]["graph"])
        return head + extra

    @staticmethod
    def _hard_names(names: list[str], graph_hits: dict) -> list[str]:
        """复杂模式补全用的具体实体：排除词典中的概念，以及命中文档占比超过
        ENTITY_SOFT_RATIO 的高频泛概念（这两类只参与排序加分，不触发整篇补全）。"""
        dic = get_dictionary()
        by_doc = {it["doc_name"]: (it.get("entities") or "") for it in graph_hits.values()}
        total = max(1, len({f.name for f in SETTINGS.raw_dir.glob("*.md")}))
        hard = []
        for n in names:
            if not n or dic.get(n, "entity") == "concept":
                continue
            if sum(1 for e in by_doc.values() if n in e) > total * ENTITY_SOFT_RATIO:
                continue
            hard.append(n)
        return hard

    @staticmethod
    def _mark_matched(items: list[dict], names: list[str]) -> list[dict]:
        """给结果标注实体链接（供前端画知识图谱）：文档 entities 列里命中的实体名。
        图谱通道是文档级召回，而这些切片由向量/BM25 选出，标注后才能看出「哪篇关联哪个实体」。"""
        names = list(dict.fromkeys(n for n in names if n))
        for it in items:
            ents = it.get("entities") or ""
            it["matched"] = [n for n in names if n in ents]
        return items

    @staticmethod
    def _where(filters: dict) -> str | None:
        if not filters:
            return None
        return " AND ".join(f"{f} = {_sql(v)}" for f, v in filters.items())

    def _safe(self, kind: str, query: str, k: int, where: str | None) -> dict:
        """带元数据预过滤执行单路召回；过滤后为空则回退全库"""
        fn = self._vector if kind == "vector" else self._fts
        if not where:
            return fn(query, k)
        hits = fn(query, k, where)
        return hits or fn(query, k)

    def _rrf(self, *channels: dict) -> list[dict]:
        merged, rrf = {}, {}
        for hits in channels:  # dict 顺序即各自排名
            for rank, (hid, item) in enumerate(hits.items()):
                if hid in merged:  # 同一切片被多路命中：合并得分，便于展示命中通道
                    merged[hid]["scores"].update(item["scores"])
                else:
                    merged[hid] = item
                rrf[hid] = rrf.get(hid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for hid, s in rrf.items():
            merged[hid]["scores"]["rrf"] = round(s, 5)
        return sorted(merged.values(), key=lambda x: -x["scores"]["rrf"])

    # ---- 入口 ----
    @staticmethod
    def _entity_names(route_info: dict) -> list[str]:
        return route_info.get("entity_keywords") or [e["name"] for e in
                                                     route_info.get("entities", [])]

    def _entity_result(self, query: str, route_info: dict, t0: float,
                       fallback_if_soft: bool = True) -> dict | None:
        """实体扫描模式：不做向量/BM25 召回，命中文件全部进结果（不设篇数上限）。

        关键词全是泛概念(概念/高命中文档占比)时返回 None，由调用方退回向量+BM25
        （单个高频概念没有可用的约束词，扫出来的几十篇基本都是噪声）。
        手动选「实体扫描」时不降级，尊重用户的显式选择。"""
        names = self._entity_names(route_info) or [query.strip()]
        sc = self._entity_scan(names)
        if fallback_if_soft and not sc["hard"]:
            return None
        return {"results": sc["docs"], "channels": ["entity"], "scan": sc["note"],
                "timings": {"search_ms": round((time.time() - t0) * 1000)}}

    def search(self, query: str, mode: str = "auto", top_k: int | None = None,
               route_info: dict | None = None) -> dict:
        k = top_k or SETTINGS.top_k
        t0 = time.time()
        route_info = route_info or {}
        channels: list[str] = []

        if mode == "vector":
            items = list(self._vector(query, k).values())
            channels = ["vector"]
        elif mode == "fts":
            items = list(self._fts(query, k).values())
            channels = ["fts"]
        elif mode == "hybrid":
            items = self._rrf(self._vector(query, k), self._fts(query, k))
            channels = ["vector", "fts"]
        elif mode == "entity":   # 手动选择：不降级
            return self._entity_result(query, route_info, t0, fallback_if_soft=False)
        else:  # auto: 按路由结果分档
            if route_info.get("level") == "entity":
                res = self._entity_result(query, route_info, t0)
                if res:
                    return res
                # 关键词全是泛概念 -> 不触发实体扫描，退回简单模式(向量+BM25+元数据)
                route_info["level"] = "simple"
                route_info["reason"] = "泛概念关键词无可约束实体，退回向量+BM25"
            complex_q = route_info.get("level") == "complex"
            where = self._where(route_info.get("filters") or {})
            ent_names = self._entity_names(route_info)
            if complex_q:
                v = self._safe("vector", query, k, where)
                f = self._safe("fts", query, k, where)
                g = self._graph(ent_names, k, where) if ent_names else {}
                items = self._dedup_docs(self._rrf(v, f, g))  # 同一文档只保留融合分最高的切片
                channels = [c for c, h in (("vector", v), ("fts", f), ("graph", g)) if h]
                if where:
                    channels.append("filter")
                # 补全仅针对具体实体链接的文档：概念/高频泛概念只参与上面的排序加分，
                # 避免单个泛概念把几十篇关联文档全部灌进提示词
                hard = self._hard_names(ent_names, g)
                if g and hard:
                    g_hard = {hid: it for hid, it in g.items()
                              if any(n in (it.get("entities") or "") for n in hard)}
                    items = self._complete_by_entity(items, g_hard, k)  # 具体实体链接文档全部补全，允许超过 top_k
                else:
                    items = items[:k]
                self._mark_matched(items, ent_names)  # 标注实体链接，供前端画知识图谱
                return {"results": items, "channels": channels,
                        "timings": {"search_ms": round((time.time() - t0) * 1000)}}
            else:
                v = self._safe("vector", query, k, where)
                f = self._safe("fts", query, k, where)
                items = self._rrf(v, f)
                channels = [c for c, h in (("vector", v), ("fts", f)) if h]
            if where:
                channels.append("filter")

        # 多路 RRF 负责从候选中选出 top-k，并按融合分降序排列
        return {"results": _by_score(items[:k]), "channels": channels,
                "timings": {"search_ms": round((time.time() - t0) * 1000)}}
