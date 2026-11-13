"""成员 B 的测试（feature/algo 分支）。

验收目标（手册 Day 9-11）：GP 拟合可复现模拟函数、输出均值/标准差；
EI/UCB 候选不重复不越界；两轮 mock 迭代成功。
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from rf_agent.active_learning import (
    expected_improvement,
    fit_gp,
    predict,
    select_next_batch,
    upper_confidence_bound,
)
from rf_agent.contracts import (
    MetricValue,
    SearchSpace,
    SimulationResult,
    SpaceParameter,
    SourceType,
    TaskStatus,
    DataLabel,
)
from rf_agent.space import check_constraints, lhs_unit


def true_fn(x: dict[str, float]) -> float:
    return (x["a"] - 0.3) ** 2 + (x["b"] - 0.7) ** 2


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        design_spec_id="spec-al-1",
        parameters=[
            SpaceParameter(
                name="a", unit="mm", lower=0.0, upper=1.0,
                source=SourceType.ENGINEERING_CONSTRAINT,
            ),
            SpaceParameter(
                name="b", unit="mm", lower=0.0, upper=1.0,
                source=SourceType.ENGINEERING_CONSTRAINT,
            ),
        ],
        approved_by="engineer",
    )


def make_result(task_id: str, value: float) -> SimulationResult:
    return SimulationResult(
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        metrics=[MetricValue(name="loss", value=value, unit="dB")],
        source=DataLabel.SIMULATED,
    )


@pytest.fixture
def training_data() -> tuple[list[SimulationResult], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(0)
    unit = rng.random((14, 2))
    results, params = [], {}
    for i, (xa, xb) in enumerate(unit):
        task_id = f"t{i:03d}"
        x = {"a": float(xa), "b": float(xb)}
        results.append(make_result(task_id, true_fn(x)))
        params[task_id] = x
    return results, params


def test_gp_fits_simulated_function(space, training_data):
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)
    assert model.feature_names == ["a", "b"]
    assert model.best_y == pytest.approx(min(true_fn(x) for x in params.values()))
    errors, covered = [], 0
    for task_id, x in params.items():
        mu, sd = predict(model, x)
        assert sd >= 0.0
        actual = true_fn(x)
        errors.append(abs(mu - actual))
        if mu - 2 * sd <= actual <= mu + 2 * sd:
            covered += 1
    assert np.mean(errors) < 0.02
    assert covered / len(params) >= 0.8


def test_fit_gp_rejects_insufficient_data():
    with pytest.raises(ValueError):
        fit_gp([], "loss", {})


def _unit_scaler(n_features: int) -> StandardScaler:
    return StandardScaler().fit(np.eye(n_features))


def test_ei_positive_when_prediction_better_than_best():
    from rf_agent.active_learning import GPProxy

    class FixedGP:
        @staticmethod
        def predict(X, return_std=True):
            return np.array([0.1]), np.array([0.2])

    # mu=0.1 远优于 best_y=0.5 且 sd>0 → EI 必为正
    model = GPProxy(metric_name="loss", feature_names=["a", "b"], version="v",
                    scaler=_unit_scaler(2), gp=FixedGP(), best_y=0.5)
    assert expected_improvement(model, {"a": 0.5, "b": 0.5}, best_y=0.5) > 0.0


def test_ei_zero_when_sd_zero_and_no_improvement():
    from rf_agent.active_learning import GPProxy

    class DeterministicGP:
        @staticmethod
        def predict(X, return_std=True):
            return np.array([0.9]), np.array([0.0])

    model = GPProxy(metric_name="loss", feature_names=["a"], version="v",
                    scaler=_unit_scaler(1), gp=DeterministicGP(), best_y=0.5)
    assert expected_improvement(model, {"a": 0.5}, best_y=0.5) == 0.0


def test_ucb_matches_formula(training_data):
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)
    x = {"a": 0.4, "b": 0.6}
    mu, sd = predict(model, x)
    assert upper_confidence_bound(model, x, kappa=2.0) == pytest.approx(mu + 2.0 * sd)
    assert upper_confidence_bound(model, x, kappa=0.0) == pytest.approx(mu)


@pytest.mark.parametrize("kind", ["ei", "ucb"])
def test_acquisition_respects_constraints(space, training_data, kind):
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)
    batch = select_next_batch(model, space, budget_remaining=10, batch_size=4, kind=kind)
    assert len(batch.candidates) == 4
    assert batch.needs_approval
    assert batch.model_version == model.version
    points = []
    for ac in batch.candidates:
        assert ac.acquisition_kind == kind
        assert ac.reason and "pred_std" in ac.reason
        assert ac.prediction_mean is not None and ac.prediction_std >= 0
        assert check_constraints(space, ac.candidate.x_physical)
        for v in ac.candidate.x_normalized.values():
            assert 0.0 <= v <= 1.0
        points.append(list(ac.candidate.x_normalized.values()))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            assert np.linalg.norm(np.array(points[i]) - np.array(points[j])) >= 0.05 - 1e-12


def test_batch_size_capped_by_budget(training_data, space):
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)
    batch = select_next_batch(model, space, budget_remaining=2, batch_size=4)
    assert len(batch.candidates) == 2
    empty = select_next_batch(model, space, budget_remaining=0, batch_size=4)
    assert empty.candidates == []


def test_two_round_mock_loop(space, training_data):
    """两轮闭环：拟合 → 选点 → 用真函数当作仿真结果 → 再拟合，样本内误差应不恶化。"""
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)

    def train_error(m, xs):
        return float(np.mean([abs(predict(m, x)[0] - true_fn(x)) for x in xs]))

    err_before = train_error(model, list(params.values()))
    for rnd in (1, 2):
        batch = select_next_batch(model, space, budget_remaining=10, batch_size=3,
                                  seed=rnd, kind="ei")
        assert len(batch.candidates) == 3
        for ac in batch.candidates:
            task_id = f"loop{rnd}-{ac.candidate.candidate_id}"
            x = ac.candidate.x_normalized
            results.append(make_result(task_id, true_fn(x)))
            params[task_id] = dict(x)
        model = fit_gp(results, "loss", params, seed=0)
    assert train_error(model, list(params.values())) <= err_before + 0.01


def test_select_next_batch_rejects_unknown_kind(training_data, space):
    results, params = training_data
    model = fit_gp(results, "loss", params, seed=0)
    with pytest.raises(ValueError):
        select_next_batch(model, space, budget_remaining=4, kind="poi")
