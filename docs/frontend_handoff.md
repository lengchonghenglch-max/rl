# 前端工作日志与后续交接

更新日期：2026-11-10  
范围：根目录 `ui/` 的 NiceGUI 前端及其 FastAPI / 服务层接入情况。

## 项目目标与约束

- 前端使用 NiceGUI 4.0.0，Plotly 用于交互式图表；Python 要求 3.12。
- 复用项目的 `contracts.py`、FastAPI、SQLite 和领域服务。`contracts.py` 是团队共用契约，未经团队确认不修改。
- 数据标签沿用契约的 `DataLabel`：`simulated` 表示模拟数据，`simulation` 表示真实仿真数据，`prediction` 表示模型预测数据。UI 的 `data_kind` 是展示/服务映射，不是新增领域契约字段。
- 前端放在仓库根目录 `ui/`。不要与原有 `src/rf_agent/ui/` 混淆，也不要擅自改动成员 C 负责的文件。
- 当前是 Mock 演示前端。不得把 Mock 候选点、曲线或指标描述成真实仿真结果。

## 已完成

- 建立 FastAPI app 和路由：`src/rf_agent/api/app.py`、`runs.py`、`approvals.py`、`artifacts.py`。
- 建立共享应用服务：`src/rf_agent/services/workflow.py`；NiceGUI 页面和 HTTP API 都可通过它处理任务，UI 不直接读写 SQLite。
- 建立固定 seed 的 Mock provider：`src/rf_agent/services/mock_provider.py`，用于固定格式的设计规格、证据、LHS 演示采样点和 S 参数 CSV。
- 建立根目录 NiceGUI 页面及组件：
  - 页面：设计、工程确认、采样、仿真数据、结果、运行记录、报告。
  - 组件：工作流步骤条、参数空间图、S 参数图、模型诊断图、证据卡、审批面板、文件表格、provenance 标签。
- 页面通过 NiceGUI 会话状态保存当前 `run_id`；持久任务快照和 artifact 存在 SQLite。
- 支持创建 run、批准/拒绝、生成演示采样点、上传 CSV/JSON、校验基本 schema、保存文件和 SHA-256、按 run 下载文件、导出基础 Markdown/CSV 报告。
- 增加 `scripts/demo_frontend.py` 演示脚本。
- 依赖声明已加入 `pyproject.toml` 和 `requirements.txt`：NiceGUI 固定 `4.0.0`，Plotly 限定 `<8`。

## 当前工作流

1. 设计页接收文字需求，服务创建 run 和演示用设计规格/证据，状态等待人工确认。
2. 确认页接受 approve 或 reject；批准后服务生成 24 个固定 seed 的 Mock LHS 候选点，任务进入 `sampled` / `submit`。
3. 采样页展示候选参数点。
4. 仿真数据页允许上传 CSV/JSON。当前校验 `frequency_hz`、`s11_db`、`s21_db` 字段并保存文件；上传数据尚未触发真实仿真或后续分析。
5. 运行记录、结果和报告页展示当前保存的快照、artifact/演示数据并提供基础导出。

页面到服务的主要链路是 `ui/pages/* -> workflow`。FastAPI 同时提供等价的 HTTP 操作入口，但 NiceGUI 页面目前不是通过 HTTP 请求 FastAPI 来调用这些操作。FastAPI 与 NiceGUI 由同一应用入口挂载；NiceGUI 4.0.0 的 `ui.run_with` 用于挂载，不负责启动服务器，入口随后由 Uvicorn 启动。

## 已知限制与需要接入的工作

### 真实领域流程

- **需求解析**：当前服务使用 Mock 设计模板。接入 `agent.py` / LLM 解析，并把真实解析模式、错误和待确认项呈现在 UI。仓库已有 LLM adapter，但尚未确认它已接入这套根目录 UI 服务流程。
- **知识检索**：当前证据是 Mock 文本。接入 `rag.py`，显示真实来源、页码、chunk、引用和检索分数，并保留可追溯性。
- **采样**：当前候选点由 Mock provider 生成。接入 `space.py` 的真实采样算法，并核对参数范围、约束过滤及 UI 图表结果。
- **仿真提交与轮询**：当前 `src/rf_agent/adapters/simulator.py` 的真实 `submit` / `poll` 尚未实现。需要接入目标仿真器（如 UltraEM）、提交任务、记录任务状态、轮询/恢复和错误处理。
- **结果 ingest 与分析**：上传只校验和保存文件，不会与候选点/仿真任务关联，也不解析成领域 `SimulationResult`。需要定义可靠的关联键和数据映射，再接入结果指标计算、S 参数可视化和异常提示。
- **主动学习**：接入 `active_learning.py` 的 GP、预测不确定性、EI/UCB 候选排序和下一批采样流程。预测结果必须标注 `prediction`，并与 `simulation` / `simulated` 清楚区分。
- **完整报告**：当前报告只列状态和文件名/hash。扩展为包含设计规格、证据、审批、候选点、仿真任务、指标、图表和 provenance 的持久化报告。

### 服务可靠性与体验

- 当前前端及 API 任务服务的 SQLite 表为 UI 工作流快照和 artifact 表；确认是否应统一写入项目现有 `storage.py` / `WorkflowState`，避免两套状态来源。
- 审批已具备幂等键处理，但要验证重复请求、不同 run 的键冲突和状态迁移一致性。
- 上传限制 5 MB，允许 CSV/JSON；需考虑行数限制、JSON 类型校验、恶意/超大输入、文件名和下载响应头处理。
- 长任务尚未作为后台任务执行，也没有进程重启后的真实仿真恢复与轮询。完成仿真接入时应采用持久任务状态，不要让 ASGI 请求同步等待长计算。
- 当前 run 会话状态保存 `run_id`，需验证浏览器刷新、跨页面导航和重启后的恢复体验；历史 run 可从运行记录页选择。
- 补齐 UI/API 集成测试与关键页面的浏览器实测。重点覆盖批准/拒绝、上传与下载、provenance 标签、刷新恢复和导出内容。

## 验证记录

前一轮实现代理报告的验证结果（本日志整理时未重新执行）：

- `pytest`：27 passed，6 skipped。
- Ruff：通过。
- `compileall`：通过。
- HTTP 冒烟：`/`、`/docs` 返回 200；创建 run 并审批后进入 sampled，生成 24 个候选点。
- 当时服务监听 `http://127.0.0.1:8000`，PID 27792，日志位于 `logs/frontend.out.log` 和 `logs/frontend.err.log`。服务/PID 是否仍存活需要接手时重新确认。

## 新 agent 接手建议

1. 先读本交接文档、`E:\AutoSearch\learn\RF_AI_Agent_前端TodoList.md`、`docs/architecture.md` 和 `contracts.py`，只针对计划要做的边界读相关实现文件；不要先重读整个仓库。
2. 运行 `git status --short`，保留工作区已有改动。不要覆盖成员分支或修改共享契约。
3. 先明确这次接入目标是“继续 Mock UI 完善”还是“真实领域闭环”。若做真实闭环，优先核清 `WorkflowService` 与现有 `agent.py`、`storage.py`、`rag.py`、`space.py`、`active_learning.py`、simulator adapter 的接口和状态映射，再做小步集成。
4. 每个图表、表格和导出都检查 `simulated`、`simulation`、`prediction` provenance；Mock 数据始终显式标记。
5. 完成后更新本日志与 TodoList 的状态，并记录实际执行的测试/启动结果；不要把上一轮验证结果当作本轮验证。

## 与原 TodoList 的关系

原始计划在 `E:\AutoSearch\learn\RF_AI_Agent_前端TodoList.md`。该文件仍保留原计划内容；它的勾选状态没有随代码实现自动更新。本文档是实现现状和后续工作的交接记录，后续 agent 应根据实际验证同步更新 TodoList。
