# 架构速览（详见 14 天手册）

```
自然语言需求 → LLM+JSON Schema → RAG证据 → 人工确认闸门
→ 参数空间/硬约束 → LHS/Sobol候选 → 约束过滤 → 仿真任务JSON
→ 仿真接口(Mock/UltraEM) → CSV校验/SQLite → GP均值+不确定性
→ EI/UCB候选打分 → (循环回仿真任务) → Streamlit报告
```

- 数据合同：`src/rf_agent/contracts.py`（唯一交互边界）
- A 输出 `Evidence[]`；B 接收 `DesignSpec` 输出 `SearchSpace/SampleBatch/AcquisitionBatch`；
  C 编排 `WorkflowState` 并调用二者。
- LLM 不能直接执行任意命令；硬约束不可被 LLM 绕过。
- 三类数据标签：simulated（假数据）/ simulation（真实仿真）/ prediction（模型预测）。
