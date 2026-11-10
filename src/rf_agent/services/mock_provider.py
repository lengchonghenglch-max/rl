from __future__ import annotations

import csv
import hashlib
import io
import random
import uuid
from datetime import UTC, datetime

from rf_agent.contracts import (
    Candidate,
    DataLabel,
    DesignSpec,
    Evidence,
    MetricSpec,
    ParameterSpec,
    SampleBatch,
    SamplerKind,
    SourceType,
)

SEED = 42


def design_spec(request: str) -> DesignSpec:
    return DesignSpec(
        spec_id=f"spec-{uuid.uuid4().hex[:8]}",
        topology="bandpass_filter",
        frequency_band_hz=(2.3e9, 2.5e9),
        parameters=[
            ParameterSpec(name="line_width_um", unit="um", lower=5, upper=50, nominal=20),
            ParameterSpec(name="gap_um", unit="um", lower=5, upper=40, nominal=18),
        ],
        metrics=[
            MetricSpec(
                name="insertion_loss_db", operator="<=", target=2.0, band_hz=(2.35e9, 2.45e9)
            ),
            MetricSpec(name="s11_db", operator="<=", target=-10.0, band_hz=(2.35e9, 2.45e9)),
        ],
        needs_confirmation=["基板材料与厚度未确认", "端口定义未确认"],
        raw_request=request,
    )


def evidence() -> list[Evidence]:
    return [
        Evidence(
            document_id="mock-doc-001",
            page=12,
            chunk_id="mock-doc-001#p12#c3",
            quote="[MOCK] 带通滤波器的线宽与耦合间距可作为初始调节参数。",
            score=0.91,
            source_type=SourceType.KNOWLEDGE_SUGGESTION,
            needs_confirmation=True,
            confidence=0.5,
        ),
        Evidence(
            document_id="mock-doc-002",
            page=7,
            chunk_id="mock-doc-002#p7#c1",
            quote="[MOCK] 基板材料与厚度应由工程师确认。",
            score=0.85,
            source_type=SourceType.KNOWLEDGE_SUGGESTION,
            needs_confirmation=True,
            confidence=0.5,
        ),
    ]


def sample(spec: DesignSpec, n: int = 24) -> SampleBatch:
    rng = random.Random(SEED)
    candidates = []
    for index in range(n):
        normalized = {p.name: (index + rng.random()) / n for p in spec.parameters}
        physical = {
            p.name: p.lower + normalized[p.name] * (p.upper - p.lower) for p in spec.parameters
        }
        candidates.append(
            Candidate(
                candidate_id=f"mock-{index + 1:03}",
                x_physical=physical,
                x_normalized=normalized,
                sampler=SamplerKind.LHS,
            )
        )
    return SampleBatch(
        batch_id=f"batch-{uuid.uuid4().hex[:8]}",
        design_spec_id=spec.spec_id,
        seed=SEED,
        sampler=SamplerKind.LHS,
        candidates=candidates,
    )


def artifacts() -> tuple[str, bytes, str]:
    rng = random.Random(SEED)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["frequency_hz", "s11_db", "s21_db", "data_kind"])
    for i in range(81):
        freq = 2.2e9 + i * 5e6
        resonance = (freq - 2.4e9) / 30e6
        s11 = -4.0 - 22.0 * (1 - min(abs(resonance), 1)) + rng.uniform(-0.3, 0.3)
        s21 = -1.0 - 8.0 * min(abs(resonance), 1) + rng.uniform(-0.2, 0.2)
        writer.writerow([f"{freq:.0f}", f"{s11:.3f}", f"{s21:.3f}", DataLabel.SIMULATED.value])
    return "mock_s_parameters.csv", out.getvalue().encode(), "text/csv"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def iso_now() -> str:
    return datetime.now(UTC).isoformat()
