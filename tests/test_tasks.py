"""成员 C 负责：Day 6 任务封装与 Mock 仿真器测试（全离线）。"""

import csv
import json

import pytest

from rf_agent import storage, tasks
from rf_agent.adapters import simulator
from rf_agent.contracts import (
    Candidate,
    DataLabel,
    DesignSpec,
    SampleBatch,
    SamplerKind,
    TaskStatus,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    storage.init_db(path)
    return path


def make_spec():
    return DesignSpec(
        spec_id="spec-t1",
        topology="bpf",
        frequency_band_hz=(2.4e9, 2.5e9),
        parameters=[{"name": "w_um", "unit": "um", "lower": 1.0, "upper": 10.0}],
        metrics=[{"name": "insertion_loss_db", "operator": "<=", "target": 2.0}],
    )


def make_batch(spec, feasible_flags=(True, True, False)):
    cands = [
        Candidate(
            candidate_id=f"c{i}",
            x_physical={"w_um": 1.0 + i},
            x_normalized={"w_um": 0.1 * i},
            sampler=SamplerKind.LHS,
            feasible=ok,
        )
        for i, ok in enumerate(feasible_flags)
    ]
    return SampleBatch(
        batch_id="batch-t1",
        design_spec_id=spec.spec_id,
        seed=42,
        sampler=SamplerKind.LHS,
        candidates=cands,
    )


# ---------------------------------------------------------------------------
# build_tasks
# ---------------------------------------------------------------------------


def test_build_tasks_skips_infeasible():
    spec = make_spec()
    built = tasks.build_tasks(make_batch(spec), spec)
    assert [t.task_id for t in built] == ["task-c0", "task-c1"]


def test_build_tasks_labels_and_timestamp():
    spec = make_spec()
    batch = make_batch(spec, feasible_flags=(True,))
    (task,) = tasks.build_tasks(batch, spec)
    assert task.source == DataLabel.SIMULATED  # mock 数据绝不允许标 simulation
    assert task.created_at == batch.created_at  # 幂等：取批次时间而非导出时刻
    assert task.design_id == spec.spec_id
    assert task.requested_metrics == ["insertion_loss_db"]
    assert task.parameter_values == batch.candidates[0].x_physical


# ---------------------------------------------------------------------------
# export_tasks：文件产物 + 幂等入库
# ---------------------------------------------------------------------------


def test_export_creates_csv_and_json(db, tmp_path):
    spec = make_spec()
    batch = make_batch(spec)
    out = tmp_path / "incoming"
    built, n_new, csv_path, json_path = tasks.export_tasks(batch, spec, db, out)

    assert n_new == len(built) == 2
    assert csv_path.exists() and json_path.exists()

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == tasks.CSV_FIELDS
    assert len(rows) == 2
    assert all(r["source"] == "simulated" for r in rows)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [d["task_id"] for d in data] == ["task-c0", "task-c1"]


def test_export_idempotent(db, tmp_path):
    """同批导出两次：文件逐字节一致，新任务数为 0。"""
    spec = make_spec()
    batch = make_batch(spec, feasible_flags=(True,))
    out = tmp_path / "incoming"

    _, n1, csv1, json1 = tasks.export_tasks(batch, spec, db, out)
    first_csv, first_json = csv1.read_bytes(), json1.read_bytes()

    _, n2, csv2, json2 = tasks.export_tasks(batch, spec, db, out)
    assert n1 == 1 and n2 == 0
    assert csv2.read_bytes() == first_csv
    assert json2.read_bytes() == first_json


# ---------------------------------------------------------------------------
# Mock 仿真器
# ---------------------------------------------------------------------------


def test_submit_run_id_deterministic():
    spec = make_spec()
    (task,) = tasks.build_tasks(make_batch(spec, (True,)), spec)
    assert simulator.submit(task) == simulator.submit(task) == f"run-{task.task_id}"


def test_mock_submit_simulated_and_deterministic():
    spec = make_spec()
    (task,) = tasks.build_tasks(make_batch(spec, (True,)), spec)
    r1 = simulator.mock_submit(task)
    r2 = simulator.mock_submit(task)
    assert r1.status == TaskStatus.SUCCESS
    assert r1.source == DataLabel.SIMULATED
    assert [m.name for m in r1.metrics] == task.requested_metrics
    assert all(m.unit == "dB" for m in r1.metrics)
    # 同参数同指标 -> 同数值（可复现）
    assert [m.value for m in r1.metrics] == [m.value for m in r2.metrics]
    # 不同参数 -> 数值有区分度
    other = task.model_copy(update={"parameter_values": {"w_um": 99.0}})
    r3 = simulator.mock_submit(other)
    assert [m.value for m in r3.metrics] != [m.value for m in r1.metrics]


def test_run_mock_batch_csv(tmp_path):
    spec = make_spec()
    built = tasks.build_tasks(make_batch(spec), spec)
    out = tmp_path / "simulated"
    csv_path = simulator.run_mock_batch(built, out)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["source"] == "simulated" for r in rows)
    assert all(r["status"] == "success" for r in rows)
    assert "insertion_loss_db" in rows[0]


def test_poll_not_implemented():
    with pytest.raises(NotImplementedError):
        simulator.poll("run-x")
