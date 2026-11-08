# RF AI Agent

三人团队的射频设计 AI 系统 MVP（14 天手册）：
自然语言需求 → 带引用 RAG → `DesignSpec` → 工程师确认参数空间 → LHS 采样 → 仿真（当前 Mock）→ GP → EI/UCB 下一批任务 → 报告。

> 本项目只负责 AI、数据和软件工作流，不负责射频拓扑选择与真实性能验收。
> 三类数据标签 `simulated / simulation / prediction` 严禁混淆。

## 分工与分支

| 成员 | 分支 | 负责文件（不要动别人的） | 职责 |
|---|---|---|---|
| A | `feature/rag` | `src/rf_agent/rag.py`、`tests/test_rag.py`、`data/knowledge/` | RAG 知识检索、引用、召回评测 |
| B | `feature/algo` | `src/rf_agent/space.py`、`active_learning.py`、`tests/test_space.py`、`tests/test_active_learning.py` | 参数空间、LHS/Sobol、GP、EI/UCB |
| C | `feature/app` | `src/rf_agent/agent.py`、`adapters/`、`ui/`、`storage.py`、`app.py` | 工作流状态机、百炼接入、Streamlit、入库、部署演示 |
| 共同 | 只能走 PR | `src/rf_agent/contracts.py` | 数据合同：改动必须三人同意 |

## 协作规矩（必须遵守）

1. 所有模块只通过 `contracts.py` 的 Pydantic 模型交互；**不得在自己的代码里复制或放宽字段**。
2. 每天开工先同步主分支：`git checkout 你的分支 && git pull --rebase origin main`。
3. 不直接改他人负责的目录；发现问题在群里说，由负责人自己改。
4. 队友接口没写完时，用**同签名 Mock**（函数名/输入输出格式一致），演示时必须标记 Mock。
5. `.env` 永不提交；提交前跑 `git diff --check` 和 `ruff check src tests`。
6. 提交信息格式：`feat/fix/test/docs: 描述`。

## 快速开始

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
copy .env.example .env    # 然后填入自己的 DASHSCOPE_API_KEY（不入库）
pytest -q                 # 合同测试必须全绿
streamlit run app.py      # Day 1 验收门槛
```

## 目录结构

```
src/rf_agent/
  contracts.py        # 三人共用数据合同（改它 = 改宪法）
  rag.py              # A：检索
  space.py            # B：参数空间与采样
  active_learning.py  # B：GP 与 EI/UCB
  agent.py            # C：工作流状态机
  storage.py          # C：SQLite 入库
  adapters/llm.py     # C：百炼客户端（唯一 LLM 出口）
  adapters/simulator.py # C：仿真 Adapter（当前 Mock）
  ui/pages.py         # C：Streamlit 页面
tests/                # test_contracts 为公共地基，其余按成员划分
data/knowledge|simulated|incoming/   # 知识文档 / 模拟数据 / 仿真结果导入
docs/                 # 架构与记录
app.py                # Streamlit 入口
```

## 当前状态

- [x] 骨架与合同（main）
- [ ] Day 1-7 阶段一：RAG 引用、约束过滤、DesignSpec 通过 schema（见手册每日计划）
- [ ] Day 8-14 阶段二：GP、主动学习、完整串联与演示
