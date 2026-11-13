"""成员 B 负责：GP 代理模型与 EI/UCB 主动选点（手册模块 6-7）。

分支：feature/algo
验收：GP 输出均值/标准差；EI/UCB 候选点不重复、不越界、
必须先过 SearchSpace 硬约束；每个选点理由可记录。

注意：fit_gp 相对 Day2 骨架增加 params_by_task 参数——
SimulationResult 本身不携带坐标，必须外部提供 task_id → 归一化参数 的映射。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from rf_agent.contracts import (
    AcquisitionBatch,
    AcquisitionCandidate,
    Candidate,
    SamplerKind,
    SearchSpace,
    SimulationResult,
    TaskStatus,
)
from rf_agent.space import lhs_unit, check_constraints, denormalize


def _kernel():
    """各向同性 RBF + 白噪声：小样本下比 ARD 更稳（B 组实验结论）。"""
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(5e-2, 5e1))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
    )


@dataclass
class GPProxy:
    """单指标 GP 代理模型。预测值来源标签固定为 PREDICTION。"""

    metric_name: str
    feature_names: list[str]
    version: str
    scaler: StandardScaler = field(repr=False)
    gp: GaussianProcessRegressor = field(repr=False)
    best_y: float = float("inf")


def _metric_value(result: SimulationResult, metric_name: str) -> float | None:
    if result.status != TaskStatus.SUCCESS:
        return None
    for m in result.metrics:
        if m.name == metric_name:
            return float(m.value)
    return None


def fit_gp(
    results: list[SimulationResult],
    metric_name: str,
    params_by_task: dict[str, dict[str, float]],
    seed: int = 0,
) -> GPProxy:
    """用成功仿真数据拟合 GaussianProcessRegressor。

    params_by_task: task_id → 归一化参数 dict；缺映射或无该指标的结果被跳过。
    只使用带真实 source 标签的数据由调用方（C 的工作流）保证。
    """
    xs, ys = [], []
    for r in results:
        value = _metric_value(r, metric_name)
        if value is None or r.task_id not in params_by_task:
            continue
        xs.append(params_by_task[r.task_id])
        ys.append(value)
    if len(ys) < 2:
        raise ValueError(f"拟合 {metric_name} 至少需要 2 个有效样本，当前 {len(ys)} 个")
    feature_names = sorted(xs[0])
    for x in xs:
        if sorted(x) != feature_names:
            raise ValueError("所有样本的参数维度必须一致")
    X = np.array([[x[k] for k in feature_names] for x in xs], dtype=float)
    scaler = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(
        kernel=_kernel(),
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    ).fit(scaler.transform(X), np.array(ys, dtype=float))
    return GPProxy(
        metric_name=metric_name,
        feature_names=feature_names,
        version=f"gp-{metric_name}-{uuid.uuid4().hex[:8]}",
        scaler=scaler,
        gp=gp,
        best_y=min(ys),
    )


def predict(model: GPProxy, x_normalized: dict[str, float]) -> tuple[float, float]:
    """返回 (均值, 标准差)。预测值来源标签固定为 PREDICTION。"""
    X = np.array([[x_normalized[k] for k in model.feature_names]], dtype=float)
    mu, sd = model.gp.predict(model.scaler.transform(X), return_std=True)
    return float(mu[0]), float(sd[0])


def expected_improvement(model: GPProxy, x: dict[str, float], best_y: float) -> float:
    """EI 采集函数（最小化目标：改进 = best_y - f(x)）。"""
    from scipy.stats import norm

    mu, sd = predict(model, x)
    if sd <= 1e-12:
        return max(0.0, best_y - mu)
    z = (best_y - mu) / sd
    return max(0.0, (best_y - mu) * norm.cdf(z) + sd * norm.pdf(z))


def upper_confidence_bound(model: GPProxy, x: dict[str, float], kappa: float = 2.0) -> float:
    """UCB = mu + kappa * sigma。用于最大化收益（如 margin）的场景。"""
    mu, sd = predict(model, x)
    return mu + kappa * sd


def select_next_batch(
    model: GPProxy,
    space: SearchSpace,
    budget_remaining: int,
    batch_size: int = 4,
    kind: str = "ei",
    seed: int = 0,
    kappa: float = 2.0,
    min_dist: float = 0.05,
) -> AcquisitionBatch:
    """生成下一批候选点：LHS 出池 → 硬约束过滤 → 采集函数打分 → 最小距离去重。"""
    if kind not in {"ei", "ucb"}:
        raise ValueError(f"kind 必须是 ei|ucb，收到 {kind!r}")
    n = max(0, min(batch_size, budget_remaining))
    candidates: list[AcquisitionCandidate] = []
    if n > 0:
        pool_size = max(64, 32 * n)
        d = len(space.parameters)
        keys = [p.name for p in space.parameters]
        scored: list[tuple[float, float, float, np.ndarray]] = []
        for row in lhs_unit(pool_size, d, seed):
            x_norm = dict(zip(keys, (float(v) for v in row), strict=True))
            if not check_constraints(space, denormalize(space, x_norm)):
                continue
            mu, sd = predict(model, x_norm)
            if kind == "ei":
                score = expected_improvement(model, x_norm, model.best_y)
            else:
                score = mu + kappa * sd
            scored.append((score, mu, sd, row))
        scored.sort(key=lambda t: -t[0])
        chosen: list[np.ndarray] = []
        for score, mu, sd, row in scored:
            if len(chosen) >= n:
                break
            if any(np.linalg.norm(row - c) < min_dist for c in chosen):
                continue
            chosen.append(row)
            x_norm = dict(zip(keys, (float(v) for v in row), strict=True))
            candidates.append(
                AcquisitionCandidate(
                    candidate=Candidate(
                        candidate_id=f"acq-{model.version}-{len(candidates):04d}",
                        x_physical=denormalize(space, x_norm),
                        x_normalized=x_norm,
                        sampler=SamplerKind.LHS,
                        feasible=True,
                    ),
                    acquisition_kind=kind,
                    score=float(score),
                    reason=(
                        f"{kind}={score:.4f}; pred_mean={mu:.4f}; pred_std={sd:.4f}; "
                        f"best_y={model.best_y:.4f}"
                    ),
                    prediction_mean=mu,
                    prediction_std=sd,
                )
            )
    return AcquisitionBatch(
        batch_id=f"ab-{uuid.uuid4().hex[:12]}",
        design_spec_id=space.design_spec_id,
        model_version=model.version,
        candidates=candidates,
        needs_approval=True,
    )
