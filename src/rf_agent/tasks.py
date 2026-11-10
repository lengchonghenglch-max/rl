"""成员 C 负责：Day 6 任务封装 API——把采样批次导出成仿真组能收的任务单。

合同（手册"三人接口与 Git 协作"）：任务字段至少含
task_id, design_id, parameter_values, requested_metrics, source, created_at。

幂等设计：
- task_id 由 candidate_id 确定性派生（task-{candidate_id}），同批重复导出 ID 不变；
- created_at 取批次的 created_at（不是导出时刻），同批导出文件内容逐字节一致；
- 入库用 INSERT OR IGNORE，重复提交不产生新记录。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rf_agent import storage
from rf_agent.contracts import DataLabel, DesignSpec, SampleBatch, SimulationTask
from rf_agent.logging_setup import get_logger

logger = get_logger("tasks")

EXPORT_DIR = Path("data/incoming")

CSV_FIELDS = [
    "task_id",
    "design_id",
    "parameter_values",
    "requested_metrics",
    "source",
    "created_at",
]


def build_tasks(batch: SampleBatch, spec: DesignSpec) -> list[SimulationTask]:
    """候选点 -> 任务列表。只导出 feasible 的点；来源标签如实标注。"""
    metric_names = [m.name for m in spec.metrics]
    tasks: list[SimulationTask] = []
    for cand in batch.candidates:
        if not cand.feasible:
            logger.info("跳过不可行候选点 %s", cand.candidate_id)
            continue
        tasks.append(
            SimulationTask(
                task_id=f"task-{cand.candidate_id}",
                design_id=spec.spec_id,
                parameter_values=cand.x_physical,
                requested_metrics=metric_names,
                # mock 采样批一律 simulated；B 的真实采样器接入后按批次来源如实标注
                source=DataLabel.SIMULATED,
                created_at=batch.created_at,
            )
        )
    return tasks


def export_tasks(
    batch: SampleBatch,
    spec: DesignSpec,
    db_path: Path = storage.DEFAULT_DB,
    out_dir: Path | None = None,
) -> tuple[list[SimulationTask], int, Path, Path]:
    """导出 CSV + JSON 任务单并入库。

    返回 (任务列表, 新任务数, csv路径, json路径)。重复导出同批次：
    文件内容不变、新任务数为 0。
    """
    out = out_dir if out_dir is not None else EXPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(batch, spec)

    n_new = sum(storage.save_task(t, batch_id=batch.batch_id, db_path=db_path) for t in tasks)

    csv_path = out / f"tasks_{batch.batch_id}.csv"
    json_path = out / f"tasks_{batch.batch_id}.json"
    rows = [
        {
            "task_id": t.task_id,
            "design_id": t.design_id,
            "parameter_values": json.dumps(t.parameter_values, sort_keys=True),
            "requested_metrics": ";".join(t.requested_metrics),
            "source": t.source.value,
            "created_at": t.created_at.isoformat(),
        }
        for t in tasks
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps([t.model_dump(mode="json") for t in tasks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "任务导出完成 batch=%s 共%d个（新增%d）-> %s / %s",
        batch.batch_id,
        len(tasks),
        n_new,
        csv_path,
        json_path,
    )
    return tasks, n_new, csv_path, json_path
