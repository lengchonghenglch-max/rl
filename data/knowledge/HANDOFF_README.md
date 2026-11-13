# RF AI Agent · RAG Store 交接包

> **一句话**：这是一个 **「论文知识库 + 仿真训练数据 + 工具脚本」的工作区**，不是 pip 包。
> 下一位拿到后，10 分钟内可以：查论文证据、打开离线网页、跑 RAG 评测、训一个 GP 基线。

## 0. 先读什么（按顺序，10 分钟）

1. 本文件；
2. `procedure/README.md` —— 12 个程序分别干什么、怎么跑；
3. `sim_data/DATA_CARD.md` —— 仿真数据的来源/口径/质量/局限；
4. `evaluation/baseline_bm25.md` —— RAG 检索基线（30 题）；
5. `KB_VERSION.json` —— 本次冻结的版本号（篇数/块数/哈希/指标）。

## 1. 工作区里有什么

这里有两条**独立的数据线**，加程序、候选、评测：

| 目录 | 内容 | 状态 |
|---|---|---|
| `paper/` | 25 篇论文 PDF（RAG 原始语料） | 25 篇 |
| `detail/` | 论文拆出的带页码 Markdown + `manifest.csv` + `index.json` | **25 篇 / 425 页 / 1,978 块**，6 个决策模块各 3 篇 + background 7 篇 |
| `detail_zh/` | 中文译文（**可选步骤**，默认不生成） | 12/25 篇（p01–p08 + a01–a04） |
| `sim_data/` | 仿真原始件与训练数据 | 64 个 s2p → 去重 30 个 run（29 可训练 + 1 理想参考） |
| `procedure/` | 12 个 Python 程序 + 网页 + 安装脚本 | 见 `procedure/README.md` |
| `evaluation/` | 30 题 RAG 评测集 + BM25 基线报告 | 见下 |
| `candidates/` | AutoSearch 检索候选记录（含被剔除的） | 18 个 CSV |
| `.autosearch/` | AutoSearch 工具链（源码 + venv，~266MB） | **不需要打包/提交**，对方用 `setup_autosearch.sh` 重建 |

**两条线怎么调**：论文线回答「依据/为什么/哪一页」，仿真线回答「这组参数会得到什么指标」；两者不混索引，统一通过 `procedure/tools_api.py` 暴露工具：

```python
from tools_api import PaperSearchTool, SurrogateTool
PaperSearchTool("../detail").search("acquisition function", k=3, module="acquisition")
SurrogateTool("../sim_data/training").predict({"c1_pf": 1.06, "c2_pf": 1.06})
```

## 2. 目录结构

```
rag store/
├── README.md                 ← 本文件（交接入口）
├── KB_VERSION.json           ← 冻结版本（篇数/块数/哈希/基线指标）
├── paper/                    25 篇 PDF
├── detail/                   英文知识库（md + manifest.csv + index.json）
├── detail_zh/                中文译文（可选）
├── sim_data/
│   ├── raw/                  原始 s2p + 截图；_archives/ 原始 zip
│   ├── processed/            runs.csv / metrics.csv / quality_flags.csv / duplicates.csv
│   ├── training/             dataset_v0.csv / splits.json / schema.json / scaler.json / baseline_report.md
│   └── DATA_CARD.md          仿真数据卡
├── procedure/                程序（见 procedure/README.md）
├── candidates/               AutoSearch 候选记录
├── evaluation/               30 题评测集 + 基线报告
└── .autosearch/              AutoSearch 工具链（可重建，不随包分发）
```

## 3. 环境准备（一次性，约 5 分钟）

```bash
cd "rag store"
python3.12 -m venv .venv                     # 需要 Python 3.12
.venv/bin/pip install -U pip
.venv/bin/pip install pymupdf streamlit pandas numpy scikit-learn
```

> 下文用 `<PY>` 指代 `.venv/bin/python`。
> 旧电脑上的 venv 路径不要硬编码；所有脚本都接受当前目录下的相对路径。

**AutoSearch（可选，只影响「搜索论文」功能）**：

```bash
bash procedure/setup_autosearch.sh   # clone 源码 + 建 .autosearch/venv + 锁版本依赖（约 300MB）
python3 procedure/search_papers.py --check
```

如果不用联网搜论文，这一步可以跳过——知识库、评测、网页、训练都不依赖它。

## 4. 10 分钟上手

```bash
# ① 打开离线网页（双击即可，中英切换）
open procedure/knowledge_base_web.html

# ② 查知识库 / 看规模
python3 procedure/tools_api.py stats
python3 procedure/tools_api.py search "constrained bayesian optimization" --module constraint -k 3

# ③ 跑 RAG 评测（30 题，BM25 基线）
python3 procedure/run_eval.py

# ④ 用代理模型预测
<PY> procedure/tools_api.py predict --params '{"c1_pf":1.06,"c2_pf":1.06}'

# ⑤ 训仿真数据基线
<PY> procedure/train_sim_baseline.py
```

## 5. 当前版本状态（2026-11-12 冻结）

| 项 | 值 |
|---|---|
| RAG 语料 | 25 篇 / 425 页 / 1,978 块；模块：6 个决策模块各 3 + background 7 |
| 论文许可 | 8 篇 CC BY 4.0 + 1 篇 CC BY-NC-SA 3.0 + 16 篇 arXiv「待核对」 |
| 中文译文 | 12/25（可选步骤；新增论文默认不汉化） |
| RAG 基线（BM25） | Recall@1/3/5 = **0.481 / 0.741 / 0.815**；MRR@5 0.611；页码正确率@1 **0.481**；文档 Recall@5 **1.000**；拒答 3/3 |
| 仿真数据 | 64 s2p → 30 unique（24 重复）→ 29 可训练 + 1 理想参考；dataset_v0 29 行（train 21 / val 4 / test 4） |
| 代理模型基线 | GP：IL R² 0.85、ripple 0.85、stop 0.75、f3db 0.62；RL 不稳定（样本少） |
| 关键限制 | BM25 单路检索；页码精度是短板；仿真只有 29 个可训练样本；部分许可待核对 |

详细数字：`KB_VERSION.json`；评测逐题：`evaluation/latest_report.md`。

## 6. 怎么扩展

- **加论文**：`python3 procedure/search_papers.py --download`（搜 + 下载 OA）→ `<PY> procedure/build_knowledge_base.py ingest --src paper --dst detail` → `python3 procedure/build_web_viewer.py`；汉化是可选的（`translate_papers.py`）。
- **加仿真数据**：新 s2p 按 `sim_data/raw/<batch>/<文件夹>/` 放好 → `python3 procedure/collect_sim_data.py`（自动去重/校验/打标签/划分）→ `<PY> procedure/train_sim_baseline.py`。
- **改检索 / 换 embedding**：改完先跑 `python3 procedure/run_eval.py`，与 `evaluation/baseline_bm25.md` 对比再决定是否上线。
- **冻结新版本**：`python3 procedure/make_handoff.py`（会更新 `KB_VERSION.json` 并打包）。

## 7. 怎么把这个库交给下一位

**方式 A（最省事）：直接发交接包**

```bash
python3 procedure/make_handoff.py
# → handoff/rag_store_handoff_<日期>.zip（约 50MB，含除 venv/会话导出外的全部文件）
# → handoff/MANIFEST.sha256
```
对方解压后按 §3 建环境、§4 跑一遍即可；AutoSearch 可选重建。

**方式 B：GitHub 私有仓库**

```bash
git init && git add . && git commit -m "rag store handoff v2026-11-12"
git remote add origin git@github.com:<account>/rag-store.git && git push -u origin main
```
- 仓库里的 `.gitignore` 已排除 `.autosearch/venv`、`_session_import/`、`__pycache__`、`handoff/`；
- 若不想把 40MB PDF 提交到公开仓库：把 `paper/` 从 git 排除，PDF 单独用网盘/zip 给；
- **不要**把 `.autosearch/venv` 提交（260MB，且依赖本机路径），让对方跑 `setup_autosearch.sh`。

**方式 C：网盘 / U 盘 / 企业微信**

把 `handoff/*.zip` + `MANIFEST.sha256` 一起发过去，附一句话：“解压后先读 README.md”。

**GitHub 协作者注意**：添加 collaborator 或让对方 fork，不要共享账号密码；私有仓库访问用 Personal Access Token（PAT），并开 2FA。

## 8. 待办 / 已知限制（交给下一位继续）

- [ ] **补论文许可**：17 篇 arXiv「待核对」逐篇确认；CC BY-NC-SA 那篇不能商用/训练；
- [ ] **页码级检索提升**：当前文档 Recall@5=1.0 但页码正确率仅 0.48，优先做按节切块/heading 重排/混合检索；
- [ ] **评测集扩到 40–50 题**，增加中文 query 观察项；
- [ ] **仿真补到 100–300 个 run**（L、线宽/间距、层叠），再做 GP 校准/不确定度/多保真；
- [ ] **答案级评测**（引用是否支撑答案/拒答措辞）目前还没做；
- [ ] **AutoSearch 依赖锁版本**：`mcp==1.27.0` / `fastmcp==3.2.4` / `paper-search-mcp==0.1.3`，不要随手升级。

## 9. 安全与合规

- 不要在聊天/文档里放 GitHub 密码；用 PAT；`_session_import/`（本会话导出）**不要**随包发出去；
- 论文许可：CC BY 可再分发（需署名），CC BY-NC-SA 仅非商业、相同方式共享；arXiv 论文逐篇确认；
- 仿真数据来源标签统一 `data_origin = simulation`，不能当实测结果；
- `paper/` 里 PDF 的再分发请遵守各篇许可。
