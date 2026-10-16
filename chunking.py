"""原始文档解析与章节切片

文件名规则: 客户_项目类型_文档类型_YYYYMMDD.md
切片策略: chunk_size=2000（字符）。整篇不超过 2000 的文档只产出一个切块；
超出 2000 的文档按 H2/H3 章节边界贪心打包成多个切块，每块都不超过 2000（超长小节按段落二次切分）。
"""
import re

_BR = re.compile(r"\[\[|\]\]")          # 剥离 wiki 双链标记
_H = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CHUNK_SIZE = 2000                        # 单个切块最大字符数


def clean_line(line: str) -> str:
    return _BR.sub("", line.rstrip())


def parse_filename(stem: str):
    """-> (客户, 项目类型, 文档类型, 日期)"""
    parts = stem.split("_")
    if len(parts) >= 4:
        return parts[0], "_".join(parts[1:-2]), parts[-2], parts[-1]
    return stem, "", "", ""


def _body(lines) -> str:
    return "\n".join(l for l in lines if l.strip() and l.strip() != "---").strip()


def _join(names, limit: int = 3) -> str:
    return " / ".join(names[:limit]) + ("…" if len(names) > limit else "")


def _split_long(body: str, max_chars: int):
    """单块超长时按段落硬切，保证每段不超过 max_chars"""
    if len(body) <= max_chars:
        return [body]
    paras, out, cur = body.split("\n\n"), [], ""
    for p in paras:
        while len(p) > max_chars:  # 单段超长(如大表格)硬切
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + len(p) + 2 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _blocks(text: str):
    """-> (标题, [(h2, [h3...], 正文), ...]) 按 H1/H2/H3 切出原子块"""
    title, cur_h2, cur_h3, buf, out = "", None, None, [], []

    def flush():
        nonlocal buf
        b = _body(buf)
        if b:
            out.append((cur_h2, [cur_h3] if cur_h3 else [], b))
        buf = []

    for line in map(clean_line, text.splitlines()):
        m = _H.match(line)
        if m:
            lvl, name = len(m.group(1)), m.group(2)
            if lvl == 1:
                if not title:
                    title = name
                continue
            flush()
            if lvl == 2:
                cur_h2, cur_h3 = name, None
            else:
                cur_h3 = name
        else:
            buf.append(line)
    flush()
    return title, out


def chunk_doc(text: str, max_chars: int = CHUNK_SIZE):
    """-> (文档标题, [{"section","subsection","text"}, ...])

    整篇 <= max_chars: 一个切块(章节标为「全文」)；
    超出: 按章节边界贪心打包成多个切块，每块 <= max_chars。
    """
    title, blocks = _blocks(text)
    if not blocks:
        return title, []
    total = sum(len(b) + 2 for _, _, b in blocks)

    if total <= max_chars:
        h2s = [h2 for h2, _, _ in blocks if h2]
        body = "\n\n".join(b for _, _, b in blocks)
        return title, [{"section": "全文", "subsection": _join(h2s), "text": body}]

    chunks, buf, subs, cur_sec, size = [], [], [], None, 0

    def flush():
        nonlocal buf, subs, cur_sec, size
        if buf:
            chunks.append({"section": cur_sec, "subsection": _join(list(dict.fromkeys(subs))),
                           "text": "\n\n".join(buf)})
        buf, subs, cur_sec, size = [], [], None, 0

    for h2, h3s, body in blocks:
        for piece in _split_long(body, max_chars):
            if buf and size + len(piece) + 2 > max_chars:
                flush()
            if cur_sec is None:
                cur_sec = h2 or "文档信息"   # 首个 H2 之前的内容(含元信息表)
            buf.append(piece)
            subs.extend(h3s)
            size += len(piece) + (2 if len(buf) > 1 else 0)
    flush()
    return title, chunks