"""成员 B 负责：参数空间与初始采样（手册模块 4）。

分支：feature/algo
验收：样本每维覆盖合理、seed 可复现、全部满足边界与约束、
物理值与归一化值可互相还原。
"""

from __future__ import annotations

import uuid

import numpy as np

from rf_agent.contracts import (
    Candidate,
    DesignSpec,
    SampleBatch,
    SamplerKind,
    SearchSpace,
    SourceType,
    SpaceParameter,
)


def build_search_space(
    spec: DesignSpec,
    sources: dict[str, SourceType] | None = None,
) -> SearchSpace:
    """从 DesignSpec 生成参数空间；未确认的范围默认按工程硬约束标注。"""
    sources = sources or {}
    parameters = [
        SpaceParameter(
            name=p.name,
            unit=p.unit,
            lower=p.lower,
            upper=p.upper,
            source=sources.get(p.name, SourceType.ENGINEERING_CONSTRAINT),
        )
        for p in spec.parameters
    ]
    return SearchSpace(design_spec_id=spec.spec_id, parameters=parameters)


def check_constraints(space: SearchSpace, x_physical: dict[str, float]) -> bool:
    """约束可行性过滤器。返回 False 的点必须被丢弃。"""
    for p in space.parameters:
        if p.name not in x_physical:
            return False
        if not (p.lower <= float(x_physical[p.name]) <= p.upper):
            return False
    return True


def normalize(space: SearchSpace, x_physical: dict[str, float]) -> dict[str, float]:
    return {
        p.name: (float(x_physical[p.name]) - p.lower) / (p.upper - p.lower)
        for p in space.parameters
    }


def denormalize(space: SearchSpace, x_normalized: dict[str, float]) -> dict[str, float]:
    return {
        p.name: p.lower + float(x_normalized[p.name]) * (p.upper - p.lower)
        for p in space.parameters
    }


def _require_approval(space: SearchSpace) -> None:
    if not space.approved_by:
        raise PermissionError("SearchSpace 未经工程师确认（approved_by 为空），禁止采样")


def lhs_unit(n: int, d: int, seed: int) -> np.ndarray:
    """拉丁超立方（自实现，不依赖 scipy）：每维分层取点加抖动。"""
    rng = np.random.default_rng(seed)
    strata = np.argsort(rng.random((n, d)), axis=0)
    jitter = rng.random((n, d))
    return (strata + jitter) / n


def _sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def _build_batch(
    space: SearchSpace,
    unit: np.ndarray,
    seed: int,
    sampler: SamplerKind,
) -> SampleBatch:
    names = [p.name for p in space.parameters]
    batch_id = f"sb-{uuid.uuid4().hex[:12]}"
    candidates: list[Candidate] = []
    seen: set[tuple[float, ...]] = set()
    for row in unit:
        x_norm = {name: float(v) for name, v in zip(names, row, strict=True)}
        x_phys = denormalize(space, x_norm)
        if not check_constraints(space, x_phys):
            continue
        key = tuple(round(v, 9) for v in row)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                candidate_id=f"{batch_id}-{len(candidates):04d}",
                x_physical=x_phys,
                x_normalized=x_norm,
                sampler=sampler,
                feasible=True,
            )
        )
    return SampleBatch(
        batch_id=batch_id,
        design_spec_id=space.design_spec_id,
        seed=seed,
        sampler=sampler,
        candidates=candidates,
    )


def sample_lhs(space: SearchSpace, n: int, seed: int) -> SampleBatch:
    """拉丁超立方采样：[0,1]^d 撒点 → 映射物理值 → 约束过滤 → 批内去重。"""
    _require_approval(space)
    if n < 1:
        raise ValueError(f"n 必须 ≥ 1，收到 {n}")
    d = len(space.parameters)
    return _build_batch(space, lhs_unit(n, d, seed), seed, SamplerKind.LHS)


def sample_sobol(space: SearchSpace, n: int, seed: int) -> SampleBatch:
    """Sobol 低差异序列采样（高维均匀覆盖优于 LHS 时使用）。"""
    _require_approval(space)
    if n < 1:
        raise ValueError(f"n 必须 ≥ 1，收到 {n}")
    d = len(space.parameters)
    return _build_batch(space, _sobol_unit(n, d, seed), seed, SamplerKind.SOBOL)


def coverage_report(batch: SampleBatch, space: SearchSpace) -> dict[str, float]:
    """每维覆盖率（样本张成的归一化跨度）+ 全局最小覆盖与重复点统计。"""
    if not batch.candidates:
        return {p.name: 0.0 for p in space.parameters} | {
            "min_coverage": 0.0,
            "duplicates": 0.0,
        }
    keys = [p.name for p in space.parameters]
    report: dict[str, float] = {}
    seen: set[tuple[float, ...]] = set()
    duplicates = 0
    for c in batch.candidates:
        point = tuple(round(c.x_normalized[k], 9) for k in keys)
        if point in seen:
            duplicates += 1
        seen.add(point)
    for k in keys:
        vals = [c.x_normalized[k] for c in batch.candidates]
        report[k] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    report["min_coverage"] = min(report[k] for k in keys)
    report["duplicates"] = float(duplicates)
    return report
