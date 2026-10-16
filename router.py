"""查询路由器：按难易程度决定检索通道

- 简单（单一事实/单文档）: 向量 + BM25 + 元数据过滤
- 复杂（跨文档聚合/对比/列表/多跳）: 额外叠加图谱召回（Wiki 词典 + Raw 双链实体倒排）
- 实体（LLM 抽出的关键词是实体或概念）: 按实体名扫描 Raw 全部文件，列出内容含 [[实体名]] 的文件，
  整篇塞进提示词，跳过关键词检索与向量检索

判定策略: 聚合/对比类模式词直接判复杂（零延迟）；否则调用 LLM 抽关键词并分类，
关键词命中 Wiki 词典（实体/概念）即判实体；LLM 失败退回简单。
"""
import json
import re

from config import SETTINGS, llm_ready
from wiki_graph import extract_entities, get_dictionary

_COMPLEX = re.compile(r"哪些|所有|全部|列出|列表|汇总|统计|分别|对比|相比|多少个|几个|每种|每个|参与|负责")
_FILTER_FIELD = {"customer": "customer", "project_type": "project_type"}


def dedup_keywords(names: list[str]) -> list[str]:
    """去掉被更长关键词包含的短词（如 项目管理系统 > 项目管理、微服务架构 > 微服务）"""
    out: list[str] = []
    for n in sorted(dict.fromkeys(names), key=len, reverse=True):
        if not any(n != o and n in o for o in out):
            out.append(n)
    return out


def _llm_route(query: str, entity_names: list[str]) -> dict:
    """LLM 一次调用：判定检索难度 + 抽取关键词 -> {level, reason, keywords}"""
    from openai import OpenAI
    client = OpenAI(base_url=SETTINGS.llm_base_url, api_key=SETTINGS.llm_api_key,
                    timeout=8, max_retries=1)
    prompt = (
        "你是企业项目文档知识库的检索路由器。文档按 客户_项目类型_文档类型_日期 组织。\n"
        "任务一 判断问题的检索难度：\n"
        "- simple: 单一事实查询，答案通常在一个文档内（预算/日期/负责人/技术选型/指标数值等）\n"
        "- complex: 跨文档聚合或列表汇总（如\"XX做过哪些项目\"\"XX负责哪些文档\"）、多实体对比、多跳关联\n"
        "任务二 抽取问题中的关键词：只保留客户企业/项目系统/技术组件/人员/部门等实体或概念名称，"
        "不要保留\"预算\"\"多少\"\"负责\"这类通用词。\n"
        f"问题：{query}\n"
        f"已识别实体：{'、'.join(entity_names) if entity_names else '无'}\n"
        '仅输出JSON：{"level":"simple或complex","reason":"不超过15字","keywords":["关键词1","关键词2"]}'
    )
    resp = client.chat.completions.create(model=SETTINGS.llm_model,
                                          messages=[{"role": "user", "content": prompt}],
                                          temperature=0, max_tokens=150)
    m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
    obj = json.loads(m.group(0)) if m else {}
    level = obj.get("level") if obj.get("level") in ("simple", "complex") else "simple"
    keywords = [str(k).strip() for k in (obj.get("keywords") or []) if str(k).strip()]
    return {"level": level, "reason": str(obj.get("reason", ""))[:30], "keywords": keywords}


def llm_keywords(query: str) -> list[str]:
    """仅抽取关键词（手动选「实体全文」模式时用），LLM 不可用或失败返回 []"""
    if not llm_ready():
        return []
    try:
        return _llm_route(query, [])["keywords"]
    except Exception:
        return []


def resolve_entities(query: str) -> list[dict]:
    """实体全文模式：LLM 关键词 + 词典匹配，保留其中全部实体/概念名（已做包含去重）"""
    dic = get_dictionary()
    names = dedup_keywords(list(extract_entities(query, dic)) + llm_keywords(query))
    return [{"name": n, "type": dic[n]} for n in names if n in dic]


def route(query: str) -> dict:
    """-> {level, reason, router, entities, entity_keywords, filters}"""
    dic = get_dictionary()
    entities = [{"name": n, "type": dic[n]}
                for n in dedup_keywords(extract_entities(query, dic))]
    keywords: list[str] = []

    if _COMPLEX.search(query):  # 聚合/对比/列表类问题优先走图谱三路召回
        level, reason, by = "complex", "命中聚合/对比/列表模式词", "pattern"
    elif llm_ready():
        try:
            r = _llm_route(query, [e["name"] for e in entities])
            level, reason, keywords, by = r["level"], r["reason"], r["keywords"], "llm"
        except Exception:
            level, reason, by = "simple", "LLM 判定失败，默认简单", "fallback"
    else:
        level, reason, by = "simple", "无模式词且 LLM 不可用", "fallback"

    # LLM 关键词命中 Wiki 词典 -> 是实体/概念，走实体全文扫描（跳过关键词与向量检索）
    ent_names = [n for n in dict.fromkeys(keywords + [e["name"] for e in entities]) if n in dic]
    ent_names = dedup_keywords(ent_names)  # 项目管理系统 > 项目管理，避免重复扫描
    if by != "pattern" and ent_names:
        level = "entity"
        reason = f"关键词命中实体/概念：{'、'.join(ent_names[:3])}"

    filters = {}
    for etype, field in _FILTER_FIELD.items():
        vals = {e["name"] for e in entities if e["type"] == etype}
        if len(vals) == 1:  # 唯一客户/项目 -> 元数据预过滤；多个（对比场景）不过滤
            filters[field] = next(iter(vals))

    return {"level": level, "reason": reason, "router": by,
            "entities": entities, "entity_keywords": ent_names, "filters": filters}