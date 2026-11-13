"""成员 B 的测试（feature/algo 分支）。

验收目标（手册 Day 5-7）：LHS seed 可复现、覆盖率、约束过滤、批内去重、
物理值与归一化值可互相还原。
"""

import pytest

from rf_agent.contracts import (
    DesignSpec,
    MetricSpec,
    ParameterSpec,
    SearchSpace,
    SpaceParameter,
    SourceType,
)
from rf_agent.space import (
    build_search_space,
    check_constraints,
    coverage_report,
    denormalize,
    normalize,
    sample_lhs,
    sample_sobol,
)


@pytest.fixture
def spec() -> DesignSpec:
    return DesignSpec(
        spec_id="spec-space-1",
        topology="lpf3",
        frequency_band_hz=(0.1e9, 8e9),
        parameters=[
            ParameterSpec(name="c1_len", unit="um", lower=10.0, upper=40.0),
            ParameterSpec(name="l1_rin", unit="um", lower=20.0, upper=80.0),
        ],
        metrics=[MetricSpec(name="s11_db", operator="<=", target=-10.0)],
    )


@pytest.fixture
def approved_space(spec: DesignSpec) -> SearchSpace:
    space = build_search_space(spec)
    space.approved_by = "engineer"
    return space


def test_build_search_space_copies_bounds(spec: DesignSpec):
    space = build_search_space(spec)
    assert space.design_spec_id == spec.spec_id
    assert space.approved_by is None
    by_name = {p.name: p for p in space.parameters}
    assert by_name["c1_len"].lower == 10.0
    assert by_name["l1_rin"].upper == 80.0
    assert all(p.source == SourceType.ENGINEERING_CONSTRAINT for p in space.parameters)


def test_sampling_requires_approval(spec: DesignSpec):
    space = build_search_space(spec)
    with pytest.raises(PermissionError):
        sample_lhs(space, 8, seed=1)


def test_lhs_reproducible_with_seed(approved_space: SearchSpace):
    a = sample_lhs(approved_space, 16, seed=42)
    b = sample_lhs(approved_space, 16, seed=42)
    c = sample_lhs(approved_space, 16, seed=43)
    assert [x.x_physical for x in a.candidates] == [x.x_physical for x in b.candidates]
    assert [x.x_physical for x in a.candidates] != [x.x_physical for x in c.candidates]


def test_all_candidates_within_bounds(approved_space: SearchSpace):
    batch = sample_lhs(approved_space, 32, seed=7)
    assert len(batch.candidates) == 32
    for p in approved_space.parameters:
        values = [cd.x_physical[p.name] for cd in batch.candidates]
        assert all(p.lower <= v <= p.upper for v in values)
        assert min(values) > p.lower and max(values) < p.upper + 1e-9


def test_normalize_roundtrip(approved_space: SearchSpace):
    for cd in sample_lhs(approved_space, 8, seed=3).candidates:
        assert set(cd.x_normalized) == {"c1_len", "l1_rin"}
        assert all(0.0 <= v <= 1.0 for v in cd.x_normalized.values())
        restored = denormalize(approved_space, cd.x_normalized)
        assert restored == pytest.approx(cd.x_physical)
        assert normalize(approved_space, restored) == pytest.approx(cd.x_normalized)


def test_candidate_ids_unique_in_batch(approved_space: SearchSpace):
    batch = sample_lhs(approved_space, 24, seed=11)
    ids = [cd.candidate_id for cd in batch.candidates]
    assert len(ids) == len(set(ids))


def test_check_constraints_rejects_out_of_range():
    space = SearchSpace(
        design_spec_id="s",
        parameters=[
            SpaceParameter(
                name="w", unit="um", lower=1.0, upper=5.0,
                source=SourceType.ENGINEERING_CONSTRAINT,
            )
        ],
    )
    assert check_constraints(space, {"w": 3.0})
    assert not check_constraints(space, {"w": 0.5})
    assert not check_constraints(space, {"w": 6.0})
    assert not check_constraints(space, {})


def test_sobol_within_bounds(approved_space: SearchSpace):
    batch = sample_sobol(approved_space, 16, seed=5)
    assert len(batch.candidates) == 16
    assert batch.sampler == "sobol"
    for p in approved_space.parameters:
        values = [cd.x_physical[p.name] for cd in batch.candidates]
        assert all(p.lower <= v <= p.upper for v in values)


def test_coverage_report(approved_space: SearchSpace):
    batch = sample_lhs(approved_space, 32, seed=9)
    report = coverage_report(batch, approved_space)
    assert report["min_coverage"] > 0.8
    assert report["duplicates"] == 0.0
