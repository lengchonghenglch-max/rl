"""Deterministic mock fixtures retained for tests and offline examples.

The production workflow uses the real RAG and LHS interfaces. Mock outputs must
retain SIMULATED/KNOWLEDGE_SUGGESTION labels and must not be shown as real results.
"""

from __future__ import annotations

import uuid

from rf_agent.contracts import (
    Candidate,
    DataLabel,
    Evidence,
    SampleBatch,
    SamplerKind,
    SourceType,
    WorkflowState,
)

MOCK_SEED = 42
MOCK_N_POINTS = 5


def mock_retrieve(state: WorkflowState) -> list[Evidence]:
    """假资料检索：返回两条固定证据，格式与 A 的真实接口一致。"""
    spec = state.design_spec
    topic = spec.topology if spec else "unknown"
    return [
        Evidence(
            document_id="mock-doc-001",
            page=12,
            chunk_id="mock-doc-001#p12#c3",
            quote=f"[MOCK] {topic} 的典型可调参数为几何尺寸与耦合间距（示例证据，非真实文献）",
            score=0.91,
            source_type=SourceType.KNOWLEDGE_SUGGESTION,
            needs_confirmation=True,
            confidence=0.5,
        ),
        Evidence(
            document_id="mock-doc-002",
            page=7,
            chunk_id="mock-doc-002#p7#c1",
            quote="[MOCK] 基板材料需工程师指定后才能确定参数范围（示例证据，非真实文献）",
            score=0.85,
            source_type=SourceType.KNOWLEDGE_SUGGESTION,
            needs_confirmation=True,
            confidence=0.5,
        ),
    ]


def mock_sample(state: WorkflowState) -> SampleBatch:
    """假初始采样：在参数范围内均匀取 MOCK_N_POINTS 个点（真实 LHS 由 B 实现）。

    固定 seed、点全部标 feasible=True；数据标签语义上是 SIMULATED（未仿真）。
    """
    spec = state.design_spec
    if spec is None:
        raise ValueError("mock_sample 需要先有 design_spec")
    params = spec.parameters
    candidates: list[Candidate] = []
    for i in range(MOCK_N_POINTS):
        frac = (i + 0.5) / MOCK_N_POINTS
        x_norm = {p.name: frac for p in params}
        x_phys = {p.name: p.lower + frac * (p.upper - p.lower) for p in params}
        candidates.append(
            Candidate(
                candidate_id=f"cand-{uuid.uuid4().hex[:8]}",
                x_physical=x_phys,
                x_normalized=x_norm,
                sampler=SamplerKind.LHS,
                feasible=True,
            )
        )
    return SampleBatch(
        batch_id=f"batch-mock-{uuid.uuid4().hex[:8]}",
        design_spec_id=spec.spec_id,
        seed=MOCK_SEED,
        sampler=SamplerKind.LHS,
        candidates=candidates,
    )


def mock_simulation_result_label() -> DataLabel:
    """提醒调用方：mock 产物一律 SIMULATED。"""
    return DataLabel.SIMULATED
