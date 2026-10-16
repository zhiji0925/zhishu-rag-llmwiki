"""FastAPI 服务：/ Web 页面，/api/meta 库信息，/api/ask SSE 流式问答"""
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

from config import SETTINGS, llm_ready
from pipeline import build_prompt, stream_answer
import router
from retriever import Retriever
from wiki_graph import extract_entities, get_dictionary

app = FastAPI(title="企业知识库 RAG 问答")
STATIC = Path(__file__).parent / "static"

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    info = {
        "table_ready": False,
        "chunks": 0,
        "embedding": {"provider": SETTINGS.embedding_provider,
                      "model": SETTINGS.embedding_model,
                      "dim": SETTINGS.embedding_dim},
        "llm": {"model": SETTINGS.llm_model or None, "ready": llm_ready()},
    }
    try:
        info["chunks"] = get_retriever().table.count_rows()
        info["table_ready"] = True
    except Exception:
        pass
    return info


@app.post("/api/ask")
async def ask(request: Request):
    body = await request.json()
    query = (body.get("query") or "").strip()
    mode = body.get("mode", "auto")
    top_k = max(1, int(body.get("top_k") or SETTINGS.top_k))  # 检索条数按页面输入值
    use_llm = bool(body.get("use_llm", True))

    if not query:
        return StreamingResponse(iter([_sse({"type": "error", "message": "query 不能为空"})]),
                                 media_type="text/event-stream")

    def gen():
        try:
            t0 = time.time()
            route_info = None
            if mode == "auto":  # 智能路由：判定难度 + 实体识别 + 元数据过滤
                t_r = time.time()
                route_info = router.route(query)
                route_info["route_ms"] = round((time.time() - t_r) * 1000)
            elif mode == "entity":  # 手动实体扫描：LLM 抽关键词 + 词典匹配出实体
                ents = router.resolve_entities(query)
                route_info = {"level": "entity", "reason": "手动选择实体命中段落扫描",
                              "router": "manual", "entities": ents,
                              "entity_keywords": [e["name"] for e in ents],
                              "filters": {}}
            res = get_retriever().search(query, mode, top_k, route_info=route_info)
            if route_info:
                keywords = route_info["entities"]  # 路由器已抽取
            else:  # 手动模式：直接走词典匹配，保证「提取的关键词」始终可用
                dic = get_dictionary()
                keywords = [{"name": n, "type": dic.get(n, "entity")}
                            for n in extract_entities(query, dic)]
            system, user = build_prompt(query, res["results"])
            res["timings"]["retrieve_ms"] = round((time.time() - t0) * 1000)
            yield _sse({"type": "meta", "query": query, "mode": mode,
                        "results": res["results"], "timings": res["timings"],
                        "channels": res.get("channels", []), "scan": res.get("scan"),
                        "route": route_info, "keywords": keywords,
                        "prompt": {"system": system, "user": user}})
            if use_llm and llm_ready():
                yield _sse({"type": "llm_start"})
                t1, n = time.time(), 0
                for delta in stream_answer(system, user):
                    n += len(delta)
                    yield _sse({"type": "delta", "text": delta})
                yield _sse({"type": "llm_done",
                            "llm_ms": round((time.time() - t1) * 1000), "chars": n})
            elif use_llm:
                yield _sse({"type": "llm_skip",
                            "message": "LLM 未配置（.env 中 LLM_*），已跳过生成，仅展示检索结果与提示词"})
            yield _sse({"type": "done"})
        except Exception as e:
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream")
