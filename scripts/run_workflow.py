"""成员 C 负责：Day 3 验收脚本——记事本流水线演示。

演示四件事：
1. 一句话需求 -> parse -> retrieve -> 停在 CONFIRM 闸门（缺信息不盲跑）
2. 工程师 approve -> 闸门放行 -> sample 产出候选点 -> 停在 SUBMIT（Day 6 接入）
3. 断点恢复：从 SQLite 重新 load，证明"关了程序也能接着干"
4. 流水账：每一步都有审计事件

运行（项目根目录）：
    .venv/Scripts/python.exe scripts/run_workflow.py            # 有 key 走真 LLM
    .venv/Scripts/python.exe scripts/run_workflow.py --mock     # 强制 Mock，不花钱
    .venv/Scripts/python.exe scripts/run_workflow.py "设计一个 5.8 GHz 低通滤波器" --mock
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if "--mock" in sys.argv:
    os.environ["DASHSCOPE_API_KEY"] = ""  # 空字符串 -> has_api_key() False -> Mock 模式

from rf_agent import storage
from rf_agent.agent import approve, create_project, run_until_stopped
from rf_agent.contracts import Phase

DEFAULT_REQUEST = "在 2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB"
# 第一个非 -- 开头的命令行参数作为需求文本，例如：
#   python scripts/run_workflow.py "设计一个 5.8 GHz 的低通滤波器" --mock
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
RAW_REQUEST = _args[0] if _args else DEFAULT_REQUEST


def show(state) -> None:
    print(f"\n--- 当前站点: {state.phase.value} ---")
    if state.pending_confirmations:
        print(f"待确认 {len(state.pending_confirmations)} 项:")
        for item in state.pending_confirmations:
            print(f"  * {item}")
    if state.latest_sample_batch:
        b = state.latest_sample_batch
        print(
            f"采样批次: {b.batch_id} ({b.sampler.value}, seed={b.seed}, "
            f"{len(b.candidates)} 个候选点)"
        )
        for c in b.candidates[:3]:
            print(f"  - {c.candidate_id}: { {k: round(v, 3) for k, v in c.x_physical.items()} }")
    if state.errors:
        print(f"错误: {state.errors}")


def main() -> int:
    project_id = f"demo-{uuid.uuid4().hex[:6]}"
    print(f"=== Day 3 流水线演示 project={project_id} ===")

    print("\n[1] 创建项目并解析需求（parse -> retrieve -> confirm 闸门）...")
    state = create_project(project_id, RAW_REQUEST)
    state = run_until_stopped(state)
    show(state)
    assert state.phase == Phase.CONFIRM, "有待确认项时必须停在 CONFIRM"

    print("\n[2] 工程师批准，闸门放行（confirm -> sample -> 停在 submit）...")
    state = approve(project_id, approver="工程师小张")
    state = run_until_stopped(state)
    show(state)
    assert state.phase == Phase.SUBMIT, "采样完成后应停在 SUBMIT（Day 6 接入）"

    print("\n[3] 模拟程序重启：从 SQLite 恢复记事本...")
    restored = storage.load_state(project_id)
    print(
        f"恢复成功: project={restored.project_id} phase={restored.phase.value} "
        f"batch={restored.latest_sample_batch.batch_id if restored.latest_sample_batch else None}"
    )

    print("\n[4] 审计流水账:")
    for ev in storage.list_events(project_id):
        print(f"  [{ev['phase']}] {ev['message']}")

    print("\n=== Day 3 验收通过：闸门会停、批准会放、断电能恢复、步步有账 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
