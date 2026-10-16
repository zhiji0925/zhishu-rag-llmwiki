"""Prompt 组装 + LLM 流式生成"""
from config import SETTINGS, llm_ready

SYSTEM_PROMPT = (
    "你是企业内部项目文档知识库的问答助手。只能依据用户提供的【参考资料】回答问题：\n"
    "1. 先用一句话给出结论，再用要点展开；用自己的话归纳，不要逐行照搬原文表格；\n"
    "2. 回答中的数字、日期、金额、人名必须与资料原文一致，不得改写或编造；\n"
    "3. 引用来源处用 [文档N] 标注；\n"
    "4. 资料不足以回答时，明确说明「知识库中未找到相关信息」，不要推测。"
)


def build_prompt(query: str, results: list) -> tuple[str, str]:
    refs = []
    for i, r in enumerate(results, 1):
        sub = f" › {r['subsection']}" if r.get("subsection") else ""
        refs.append(f"[文档{i}] 《{r['doc_name']}》｜章节：{r['section']}{sub}\n{r['text']}")
    user = "【参考资料】\n" + "\n\n".join(refs) + f"\n\n【用户问题】\n{query}"
    return SYSTEM_PROMPT, user


def stream_answer(system: str, user: str):
    """流式生成器，逐段 yield 回答文本"""
    from openai import OpenAI
    client = OpenAI(base_url=SETTINGS.llm_base_url, api_key=SETTINGS.llm_api_key,
                    timeout=120, max_retries=2)
    stream = client.chat.completions.create(
        model=SETTINGS.llm_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        stream=True, temperature=0.35, max_tokens=2000,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
