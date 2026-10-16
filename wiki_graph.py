"""Wiki 实体词典与实体抽取

从 Wiki/index.md（实体索引小节）+ Wiki/entity、Wiki/concept 目录文件名构建
"实体名 -> 类型" 词典，供查询实体识别与图谱召回流使用。
类型: customer / project_type / person / dept / tech / concept / entity
"""
import re

from config import SETTINGS

_LINK = re.compile(r"\[\[([^\]\|]+?)(\|[^\]]*)?\]\]")
_SECTION_TYPE = {  # index.md 实体索引的 ### 小节 -> 类型
    "客户企业": "customer",
    "技术组件": "tech",
    "项目系统": "project_type",
    "人员": "person",
    "部门": "dept",
}
_ASCII_NAME = re.compile(r"^[A-Za-z0-9 .+#/_-]+$")

_DIC: dict[str, str] | None = None


def build_dictionary() -> dict[str, str]:
    dic: dict[str, str] = {}
    wiki = SETTINGS.raw_dir.parent / "Wiki"
    idx = wiki / "index.md"
    if idx.exists():
        h2 = cur_type = None
        for line in idx.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                h2, cur_type = line[3:].strip(), None
            elif line.startswith("### "):
                cur_type = _SECTION_TYPE.get(line[4:].strip())
            else:
                t = cur_type or ("customer" if h2 and "客户列表" in h2 else None)
                if t:
                    for m in _LINK.finditer(line):
                        name = m.group(1).strip()
                        if name:
                            dic.setdefault(name, t)
    for f in (wiki / "entity").glob("*.md"):   # 兜底：词条文件名
        dic.setdefault(f.stem, "entity")
    for f in (wiki / "concept").glob("*.md"):
        dic.setdefault(f.stem, "concept")
    return dic


def get_dictionary() -> dict[str, str]:
    global _DIC
    if _DIC is None:
        _DIC = build_dictionary()
    return _DIC


def extract_entities(text: str, dic: dict[str, str] | None = None) -> list[str]:
    """词典匹配：中文名子串匹配；ASCII 名做字母边界匹配（避免 Java 命中 JavaScript）"""
    dic = dic or get_dictionary()
    found = []
    for name in sorted(dic, key=len, reverse=True):
        if _ASCII_NAME.fullmatch(name):
            if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text):
                found.append(name)
        elif name in text:
            found.append(name)
    return found
