"""合同测试：三人共同的地基。contracts.py 的任何改动都必须先过这里。"""

import pytest
from pydantic import ValidationError

from rf_agent.contracts import (
    AcquisitionBatch,
    Budget,
    DataLabel,
    DesignSpec,
    Evidence,
    MetricSpec,
    MetricValue,
    ParameterSpec,
    Phase,
    SimulationResult,
    SimulationTask,
    SourceType,
    TaskStatus,
    WorkflowState,
)


def make_spec() -> DesignSpec:
    return DesignSpec(
        spec_id="spec-001",
        topology="bandpass_filter",
        frequency_band_hz=(2.3e9, 2.5e9),
        parameters=[ParameterSpec(name="line_width_um", unit="um", lower=1.0, upper=20.0, nominal=5.0)],
        metrics=[MetricSpec(name="insertion_loss_db", operator="<=", target=2.0)],
        raw_request="2.4 GHz 附近做一个带通滤波器，通带插损小于 2 dB",
    )


class TestDesignSpec:
    def test_valid_spec_passes(self):
        spec = make_spec()
        assert spec.parameters[0].nominal == 5.0

    def test_reversed_bounds_rejected(self):
        with pytest.raises(ValidationError):
            ParameterSpec(name="gap_um", unit="um", lower=10.0, upper=1.0)

    def test_nominal_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ParameterSpec(name="gap_um", unit="um", lower=1.0, upper=10.0, nominal=99.0)

    def test_empty_parameters_rejected(self):
        with pytest.raises(ValidationError):
            DesignSpec(
                spec_id="spec-bad",
                topology="x",
                frequency_band_hz=(1e9, 2e9),
                parameters=[],
                metrics=[MetricSpec(name="s11_db", operator="<=", target=-10.0)],
            )

    def test_bad_operator_rejected(self):
        with pytest.raises(ValidationError):
            MetricSpec(name="s11_db", operator="~=", target=-10.0)

    def test_in_band_needs_two_values(self):
        with pytest.raises(ValidationError):
            MetricSpec(name="bw", operator="in_band", target=1.0)


class TestEvidence:
    def test_knowledge_suggestion_must_need_confirmation(self):
        with pytest.raises(ValidationError):
            Evidence(
                document_id="doc-1",
                chunk_id="c-1",
                quote="...",
                source_type=SourceType.KNOWLEDGE_SUGGESTION,
                needs_confirmation=False,
            )

    def test_engineering_constraint_ok(self):
        ev = Evidence(
            document_id="doc-1",
            chunk_id="c-1",
            quote="最小间距 2um",
            source_type=SourceType.ENGINEERING_CONSTRAINT,
        )
        assert ev.needs_confirmation is False


class TestSimulationContract:
    def test_mock_task_roundtrip(self):
        task = SimulationTask(
            task_id="t-1",
            design_id="spec-001",
            parameter_values={"line_width_um": 5.0},
            requested_metrics=["insertion_loss_db"],
            source=DataLabel.SIMULATED,
        )
        result = SimulationResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCESS,
            metrics=[MetricValue(name="insertion_loss_db", value=1.5, unit="dB")],
            source=DataLabel.SIMULATED,
            simulator_version="mock-0.1",
        )
        assert result.metrics[0].value == 1.5

    def test_success_without_metrics_rejected(self):
        with pytest.raises(ValidationError):
            SimulationResult(task_id="t-2", status=TaskStatus.SUCCESS, source=DataLabel.SIMULATION)

    def test_failed_without_error_rejected(self):
        with pytest.raises(ValidationError):
            SimulationResult(task_id="t-3", status=TaskStatus.FAILED, source=DataLabel.SIMULATION)


class TestWorkflowState:
    def test_transition_updates_phase(self):
        state = WorkflowState(project_id="p-1", budget=Budget(total_simulations=20))
        state.transition(Phase.CONFIRM)
        assert state.phase == Phase.CONFIRM
        assert state.budget.remaining == 20

    def test_state_serializable_roundtrip(self):
        state = WorkflowState(project_id="p-2", design_spec=make_spec())
        restored = WorkflowState.model_validate_json(state.model_dump_json())
        assert restored.design_spec.spec_id == "spec-001"


class TestAcquisition:
    def test_batch_schema(self):
        batch = AcquisitionBatch(
            batch_id="acq-1",
            design_spec_id="spec-001",
            model_version="gp-v0",
            candidates=[],
        )
        assert batch.needs_approval is True
