"""
A 组（文献库）→ B 组（算法）的数据交接接口
==========================================
作用：B 组需要"从文献里找合理参数范围"时，不直接去搜论文，而是向 A 组的文献库
要证据（Evidence）。本文件定义这个交接的格式和三种对接方式。
三种模式，按 A 组的进度切换，B 组的其他代码一行都不用改：
    stub   本地假数据，用来开发和测试（现在就能用）
    file   A 组导出 JSON 文件，B 组读文件（最省事，推荐先用这个）
    http   A 组起服务，B 组调 HTTP 接口（最正式）
放哪里：B_Group/src/literature.py
------------------------------------------------------------------
【怎么运行】
------------------------------------------------------------------
方法一（推荐，在 B_Group 目录下）：
    cd /d D:\rf\挑战杯\B_Group
    python -m src.literature "3阶 C-L-C 低通 1.5GHz 50欧"
方法二（在 src 目录下直接跑）：
    cd /d D:\rf\挑战杯\B_Group\src
    python literature.py "3阶 C-L-C 低通 1.5GHz 50欧"
  注意：
    - 查询内容必须加英文双引号，因为里面有空格
    - 引号里那个是你要查的话，别漏
其他用法：
    python -m src.literature "..." --mode file     # 读 A 组导出的 JSON
    python -m src.literature "..." --mode http     # 调 A 组的服务
    python -m src.literature "..." --json          # 输出 JSON
    python -m src.literature "..." --k 5           # 只要 5 条
常见报错：
    error: the following arguments are required: query
        → 忘了在后面加要查的内容
    [失败] 没找到 ...\a_group_evidence
        → 用了 --mode file，但那个文件夹里还没有 A 组给的 JSON
环境变量（可选）：
    A_GROUP_MODE      默认模式 stub / file / http
    A_GROUP_RAG_URL   http 模式的接口地址
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parents[1]

DEMO_FILE = ROOT / "data" / "demo测试用数据集" / "evidence_sample.json"
FILE_INBOX = ROOT / "data" / "raw原数据" / "a_group_evidence"

DEFAULT_MODE = os.getenv("A_GROUP_MODE", "stub")
A_GROUP_URL = os.getenv("A_GROUP_RAG_URL", "http://127.0.0.1:8000/retrieve")

VALID_MODES = ("stub", "file", "http")


# ---------------------------------------------------------------- 数据契约

@dataclass
class Evidence:
    """A 组交过来的一条证据。字段名就是两边约定的交货单，别随意改。"""

    document_id: str          # 哪份文档（文件名或 A 组的文档 ID）
    chunk_id: str             # 哪一段（切块编号）
    quote: str                # 原文片段（必须是真的原文，不能是摘要）
    page: int | None = None   # 第几页，没有就 None
    score: float = 0.0        # 相关度，A 组给多少算多少
    url: str | None = None    # 原始链接
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiteratureResult:
    query: str
    mode: str
    evidence: list[Evidence]
    note: str = ""


class EvidenceFormatError(ValueError):
    """A 组给的数据不符合约定时抛这个，方便一眼看出是交接问题而不是算法问题。"""


# ---------------------------------------------------------------- 校验

def _to_evidence(item: dict[str, Any], index: int) -> Evidence:
    if not isinstance(item, dict):
        raise EvidenceFormatError(f"第 {index} 条不是对象：{type(item).__name__}")

    missing = [k for k in ("document_id", "chunk_id", "quote") if not item.get(k)]
    if missing:
        raise EvidenceFormatError(
            f"第 {index} 条缺字段 {missing}。最少要有 document_id / chunk_id / quote。"
        )

    page = item.get("page")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            raise EvidenceFormatError(f"第 {index} 条的 page 不是整数：{page!r}")

    try:
        score = float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        raise EvidenceFormatError(f"第 {index} 条的 score 不是数字：{item.get('score')!r}")

    return Evidence(
        document_id=str(item["document_id"]),
        chunk_id=str(item["chunk_id"]),
        quote=str(item["quote"]).strip(),
        page=page,
        score=score,
        url=item.get("url"),
        meta=dict(item.get("meta") or {}),
    )


def _normalize(raw: Any) -> list[dict[str, Any]]:
    """接受三种形态：{"evidence": [...]}、[...]、{"results": [...]}"""
    if isinstance(raw, dict):
        raw = raw.get("evidence") or raw.get("results") or []
    if not isinstance(raw, list):
        raise EvidenceFormatError(f"证据列表应该是数组，实际是 {type(raw).__name__}")
    return raw


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str]] = set()
    out: list[Evidence] = []
    for e in items:
        key = (e.document_id, e.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------- 三种模式

_SAMPLE = {
    "evidence": [
        {
            "document_id": "DEMO-001-placeholder",
            "chunk_id": "c001",
            "page": 12,
            "score": 0.91,
            "url": None,
            "quote": "【示例原文，不是真实文献】这里放 A 组检索到的原文片段。"
                     "比如三阶 Butterworth 低通原型的归一化元件值 g1=1.0、g2=2.0、g3=1.0。",
            "meta": {"source": "demo"},
        },
        {
            "document_id": "DEMO-002-placeholder",
            "chunk_id": "c014",
            "page": 5,
            "score": 0.77,
            "url": None,
            "quote": "【示例原文，不是真实文献】这里放片上电感/电容实现相关的经验值，"
                     "例如电感密度、Q 值范围、电容面密度等。",
            "meta": {"source": "demo"},
        },
        {
            "document_id": "DEMO-003-placeholder",
            "chunk_id": "c003",
            "page": None,
            "score": 0.64,
            "url": None,
            "quote": "【示例原文，不是真实文献】这里放同类拓扑在相近频段的实现案例，"
                     "用来做参数范围的 sanity check。",
            "meta": {"source": "demo"},
        },
    ]
}


def _ensure_demo_file() -> Path:
    """第一次运行时自动生成示例文件，省得手动准备。"""
    if not DEMO_FILE.exists():
        DEMO_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEMO_FILE.write_text(
            json.dumps(_SAMPLE, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return DEMO_FILE


def retrieve_from_stub(query: str, k: int) -> LiteratureResult:
    path = _ensure_demo_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = [_to_evidence(x, i) for i, x in enumerate(_normalize(raw))]
    items.sort(key=lambda e: e.score, reverse=True)
    return LiteratureResult(
        query=query,
        mode="stub",
        evidence=items[:k],
        note=f"假数据，来自 {path.name}（开发阶段用，不是真实文献）",
    )


def retrieve_from_files(query: str, k: int) -> LiteratureResult:
    if not FILE_INBOX.exists():
        raise FileNotFoundError(
            f"没找到 {FILE_INBOX}\n"
            f"让 A 组把检索结果导出成 JSON 放进这个文件夹，格式："
            f'{{"evidence": [{{"document_id": "...", "chunk_id": "...", "quote": "..."}}]}}'
        )

    files = sorted(FILE_INBOX.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"{FILE_INBOX} 里没有 .json 文件")

    collected: list[Evidence] = []
    for f in files:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceFormatError(f"{f.name} 不是合法 JSON：{exc}") from exc
        collected.extend(_to_evidence(x, i) for i, x in enumerate(_normalize(raw)))

    collected = _dedupe(collected)
    collected.sort(key=lambda e: e.score, reverse=True)
    return LiteratureResult(
        query=query,
        mode="file",
        evidence=collected[:k],
        note=f"读了 {len(files)} 个文件，去重后 {len(collected)} 条",
    )


def retrieve_from_http(query: str, k: int) -> LiteratureResult:
    payload = json.dumps({"query": query, "k": k}).encode("utf-8")
    req = urllib.request.Request(
        A_GROUP_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # 网络错误统一给一句人话
        raise ConnectionError(
            f"连不上 A 组的接口 {A_GROUP_URL}（{exc}）。\n"
            f"先用 --mode stub 或 --mode file 顶上。"
        ) from exc

    items = [_to_evidence(x, i) for i, x in enumerate(_normalize(raw))]
    return LiteratureResult(query=query, mode="http", evidence=items[:k], note=A_GROUP_URL)


def retrieve(query: str, k: int = 8, mode: str | None = None) -> LiteratureResult:
    """B 组统一从这里拿文献证据。切换模式不用改调用方。"""
    mode = (mode or DEFAULT_MODE).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode 只能是 {VALID_MODES}，收到 {mode!r}")

    if mode == "stub":
        return retrieve_from_stub(query, k)
    if mode == "file":
        return retrieve_from_files(query, k)
    return retrieve_from_http(query, k)


# ---------------------------------------------------------------- 命令行

def _print_human(result: LiteratureResult) -> None:
    print(f"查询：{result.query}")
    print(f"模式：{result.mode}   命中：{len(result.evidence)} 条")
    if result.note:
        print(f"说明：{result.note}")
    print("-" * 60)
    for i, e in enumerate(result.evidence, 1):
        where = f"{e.document_id}" + (f"  p.{e.page}" if e.page is not None else "")
        print(f"[{i}] {where}   score={e.score:.2f}  chunk={e.chunk_id}")
        print(f"    {e.quote}")
        if e.url:
            print(f"    {e.url}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="向 A 组文献库要证据（B 组接口）")
    parser.add_argument("query", help="要查的内容，例如：3阶 C-L-C 低通 1.5GHz 50欧")
    parser.add_argument("--mode", choices=VALID_MODES, default=None, help="对接方式")
    parser.add_argument("--k", type=int, default=8, help="要几条，默认 8")
    parser.add_argument("--json", action="store_true", help="输出 JSON，方便被别的程序调用")
    args = parser.parse_args(argv)

    try:
        result = retrieve(args.query, k=args.k, mode=args.mode)
    except (EvidenceFormatError, FileNotFoundError, ConnectionError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
