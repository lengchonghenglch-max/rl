"""三人共用的数据合同（Pydantic 模型）。

规则（来自 14 天手册）：
- 所有模块只通过本文件的模型交互，禁止私自复制或放宽字段。
- 修改本文件必须三人商量，并在 main 分支合并。
- 数据来源标签（DataLabel / SourceType）必须如实填写，
  不得把 simulated（模拟）或 prediction（预测）冒充 simulation（真实仿真）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 枚举：数据来源与类型标签
# ---------------------------------------------------------------------------

class DataLabel(str, Enum):
    """三类数据必须严格区分（手册硬性要求）。"""

    SIMULATED = "simulated"      # 我们自己编的假数据 / Mock
    SIMULATION = "simulation"    # 真实仿真器（UltraEM 等）跑出的结果
    PREDICTION = "prediction"    # GP 等代理模型的预测值


class SourceType(str, Enum):
    """参数范围的出处（关键算法连接一节）。"""

    ENGINEERING_CONSTRAINT = "engineering_constraint"  # 工程硬约束，工程师给定
    HISTORICAL_DATA = "historical_data"                # 历史项目数据（软证据）
    KNOWLEDGE_SUGGESTION = "knowledge_suggestion"      # RAG/LLM 建议，需人工确认
    UNKNOWN = "unknown"


class SamplerKind(str, Enum):
    RANDOM = "random"
    GRID = "grid"
    LHS = "lhs"
    SOBOL = "sobol"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Phase(str, Enum):
    """工作流状态机的节点（模块 3）。"""

    PARSE = "parse"
    RETRIEVE = "retrieve"
    CONFIRM = "confirm"
    SAMPLE = "sample"
    SUBMIT = "submit"
    INGEST = "ingest"
    FIT = "fit"
    ACQUIRE = "acquire"
    REPORT = "report"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# 需求规格（模块 1，A/B/C 共同上游）
# ---------------------------------------------------------------------------

class ParameterSpec(BaseModel):
    name: str
    unit: str
    lower: float
    upper: float
    nominal: float | None = None

    @field_validator("name", "unit")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name/unit 不能为空")
        return v

    @model_validator(mode="after")
    def _check_bounds(self) -> "ParameterSpec":
        if self.lower >= self.upper:
            raise ValueError(f"{self.name}: 下界 {self.lower} 必须小于上界 {self.upper}")
        if self.nominal is not None and not (self.lower <= self.nominal <= self.upper):
            raise ValueError(f"{self.name}: 标称值 {self.nominal} 超出范围")
        return self


class MetricSpec(BaseModel):
    name: str                                  # 例：s11_db, insertion_loss_db, bandwidth_hz
    operator: str                              # <=, >=, in_band
    target: float | list[float]
    band_hz: tuple[float, float] | None = None

    @field_validator("operator")
    @classmethod
    def _check_operator(cls, v: str) -> str:
        allowed = {"<=", ">=", "==", "in_band"}
        if v not in allowed:
            raise ValueError(f"operator 必须是 {allowed} 之一，收到 {v!r}")
        return v

    @model_validator(mode="after")
    def _check_target(self) -> "MetricSpec":
        if self.operator == "in_band":
            if not isinstance(self.target, list) or len(self.target) != 2:
                raise ValueError("in_band 需要 [low, high] 两个值")
        if self.band_hz is not None and self.band_hz[0] >= self.band_hz[1]:
            raise ValueError("band_hz 下界必须小于上界")
        return self


class DesignSpec(BaseModel):
    """A/C 产出，B 消费的核心规格。"""

    spec_id: str
    spec_version: int = 1
    topology: str
    frequency_band_hz: tuple[float, float]
    parameters: list[ParameterSpec]
    metrics: list[MetricSpec]
    needs_confirmation: list[str] = Field(default_factory=list)
    raw_request: str = ""                      # 工程师原始自然语言，存档用
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_band(self) -> "DesignSpec":
        if self.frequency_band_hz[0] >= self.frequency_band_hz[1]:
            raise ValueError("frequency_band_hz 下界必须小于上界")
        if not self.parameters:
            raise ValueError("至少需要一个参数")
        if not self.metrics:
            raise ValueError("至少需要一个指标")
        return self


# ---------------------------------------------------------------------------
# RAG 证据（模块 2，A 产出）
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    document_id: str
    page: int | None = None
    chunk_id: str
    quote: str
    score: float = 0.0
    source_type: SourceType = SourceType.UNKNOWN
    needs_confirmation: bool = False           # knowledge_suggestion 必须为 True
    confidence: float | None = None

    @model_validator(mode="after")
    def _check_suggestion(self) -> "Evidence":
        if self.source_type == SourceType.KNOWLEDGE_SUGGESTION and not self.needs_confirmation:
            raise ValueError("RAG/LLM 建议必须标记 needs_confirmation=True")
        return self


# ---------------------------------------------------------------------------
# 参数空间与采样（模块 4，B 产出）
# ---------------------------------------------------------------------------

class SpaceParameter(BaseModel):
    """SearchSpace 中的一维，带来源标签。"""

    name: str
    unit: str
    lower: float
    upper: float
    source: SourceType
    is_discrete: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> "SpaceParameter":
        if self.lower >= self.upper:
            raise ValueError(f"{self.name}: 下界必须小于上界")
        return self


class Constraint(BaseModel):
    """线性/关系约束：names 系数点积 <=（或 >=）bound。由 B 实现检查逻辑。"""

    name: str
    expression: str                            # 人类可读，如 "gap >= 2*line_width"
    kind: str = "custom"                       # linear | custom | drc


class SearchSpace(BaseModel):
    design_spec_id: str
    parameters: list[SpaceParameter]
    constraints: list[Constraint] = Field(default_factory=list)
    approved_by: str | None = None             # 工程师确认后才允许采样

    @model_validator(mode="after")
    def _need_approval(self) -> "SearchSpace":
        if not self.parameters:
            raise ValueError("SearchSpace 至少需要一维参数")
        return self


class Candidate(BaseModel):
    candidate_id: str
    x_physical: dict[str, float]               # 带单位的物理值
    x_normalized: dict[str, float]             # [0,1] 超立方体值
    sampler: SamplerKind
    feasible: bool = True                      # 是否通过约束过滤


class SampleBatch(BaseModel):
    batch_id: str
    design_spec_id: str
    seed: int
    sampler: SamplerKind
    candidates: list[Candidate]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _no_duplicate(self) -> "SampleBatch":
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("批内 candidate_id 不允许重复")
        return self


# ---------------------------------------------------------------------------
# 仿真任务接口（模块 5，C 封装，输入输出字段来自手册）
# ---------------------------------------------------------------------------

class SimulationTask(BaseModel):
    """提交给仿真接口（真实或 Mock）的任务。"""

    task_id: str
    design_id: str
    parameter_values: dict[str, float]
    requested_metrics: list[str]
    source: DataLabel                          # Mock 时必须为 SIMULATED
    created_at: datetime = Field(default_factory=utc_now)


class MetricValue(BaseModel):
    name: str
    value: float
    unit: str
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    extractor_version: str = "v0"


class SimulationResult(BaseModel):
    task_id: str
    status: TaskStatus
    metrics: list[MetricValue] = Field(default_factory=list)
    raw_file: str | None = None                # Touchstone/CSV 路径
    source: DataLabel                          # 真实仿真必须是 SIMULATION
    simulator_version: str = ""
    completed_at: datetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _success_needs_metrics(self) -> "SimulationResult":
        if self.status == TaskStatus.SUCCESS and not self.metrics:
            raise ValueError("status=success 时 metrics 不能为空")
        if self.status == TaskStatus.FAILED and not self.error:
            raise ValueError("status=failed 时必须给出 error 信息")
        return self


# ---------------------------------------------------------------------------
# 主动学习（模块 7，B 产出，C 调度）
# ---------------------------------------------------------------------------

class AcquisitionCandidate(BaseModel):
    candidate: Candidate
    acquisition_kind: str                      # ei | ucb
    score: float
    reason: str = ""                           # 选点理由，必须可审计
    prediction_mean: float | None = None       # 来源标签固定为 PREDICTION
    prediction_std: float | None = None


class AcquisitionBatch(BaseModel):
    batch_id: str
    design_spec_id: str
    model_version: str
    candidates: list[AcquisitionCandidate]
    needs_approval: bool = True                # 超预算前必须人工批准
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# 工作流状态（模块 3，C 编排）
# ---------------------------------------------------------------------------

class Budget(BaseModel):
    total_simulations: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total_simulations - self.used)


class WorkflowState(BaseModel):
    project_id: str
    phase: Phase = Phase.PARSE
    round_index: int = 0
    design_spec: DesignSpec | None = None
    search_space: SearchSpace | None = None
    latest_sample_batch: SampleBatch | None = None
    latest_acquisition: AcquisitionBatch | None = None
    budget: Budget | None = None
    pending_confirmations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)

    def transition(self, next_phase: Phase) -> None:
        """C 的状态机在节点间跳转时调用；记录时间便于恢复。"""
        self.phase = next_phase
        self.updated_at = utc_now()
