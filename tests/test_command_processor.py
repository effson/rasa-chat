"""Tests for CommandProcessor — applying Commands to DialogueState."""

from __future__ import annotations

import pytest

from Customer_Service_Assistant.service.schemas import (
    CanceledSystemContext,
    DialogueState,
    InterruptedSystemContext,
    ResumedSystemContext,
    StartedSystemContext,
    TaskContext,
)
from Customer_Service_Assistant.service.task import (
    CancelFlowCommand,
    CommandProcessor,
    CommandProcessorError,
    FlowLoader,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FLOW_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "flow_config"


@pytest.fixture(scope="module")
def flows():
    return FlowLoader().load([FLOW_DIR / "user_flows.yml", FLOW_DIR / "system_flows.yml"])


@pytest.fixture
def processor():
    return CommandProcessor()


@pytest.fixture
def state():
    return DialogueState(sender_id="test_user")


# ---------------------------------------------------------------------------
# start_flow
# ---------------------------------------------------------------------------


class TestStartFlow:
    def test_no_active_task_creates_task_and_system_started(
        self, processor, state, flows
    ):
        processor.run(
            [StartFlowCommand(command="start_flow", flow="refund_request")],
            state,
            flows,
        )
        assert state.active_task is not None
        assert state.active_task.flow_id == "refund_request"
        assert state.active_task.step_id == "start"

        sf = state.active_system_flow
        assert isinstance(sf, StartedSystemContext)
        assert sf.started_flow_id == "refund_request"
        assert sf.started_flow_name == "退款申请"

    def test_same_flow_already_active_is_noop(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="ask_order_number")
        state.active_system_flow = StartedSystemContext(
            started_flow_id="refund_request",
            started_flow_name="退款申请",
        )
        processor.run(
            [StartFlowCommand(command="start_flow", flow="refund_request")],
            state,
            flows,
        )
        # active_task unchanged
        assert state.active_task.flow_id == "refund_request"
        assert state.active_task.step_id == "ask_order_number"
        # system flow cleared (always cleared before check)
        assert state.active_system_flow is None

    def test_different_flow_active_pauses_and_interrupts(
        self, processor, state, flows
    ):
        state.active_task = TaskContext(flow_id="logistics_tracking", step_id="listen")
        processor.run(
            [StartFlowCommand(command="start_flow", flow="refund_request")],
            state,
            flows,
        )
        # Old task paused
        assert len(state.paused_tasks) == 1
        assert state.paused_tasks[0].flow_id == "logistics_tracking"

        # New task active
        assert state.active_task.flow_id == "refund_request"

        # System context: interrupted
        sf = state.active_system_flow
        assert isinstance(sf, InterruptedSystemContext)
        assert sf.interrupted_flow_id == "logistics_tracking"
        assert sf.started_flow_id == "refund_request"

    def test_unknown_flow_raises(self, processor, state, flows):
        with pytest.raises(CommandProcessorError, match="未知的 flow_id"):
            processor.run(
                [StartFlowCommand(command="start_flow", flow="nonexistent")],
                state,
                flows,
            )

    def test_clears_existing_system_flow(self, processor, state, flows):
        state.active_system_flow = CanceledSystemContext(
            canceled_flow_id="old", canceled_flow_name="Old"
        )
        processor.run(
            [StartFlowCommand(command="start_flow", flow="onboarding")],
            state,
            flows,
        )
        assert isinstance(state.active_system_flow, StartedSystemContext)


# ---------------------------------------------------------------------------
# set_slots
# ---------------------------------------------------------------------------


class TestSetSlots:
    def test_writes_slots_to_active_task(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [SetSlotsCommand(command="set_slots", slots={"order_number": "10086"})],
            state,
            flows,
        )
        assert state.active_task.slots["order_number"] == "10086"

    def test_writes_multiple_slots(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [
                SetSlotsCommand(
                    command="set_slots",
                    slots={"order_number": "10086", "refund_reason": "商品破损"},
                )
            ],
            state,
            flows,
        )
        assert state.active_task.slots["order_number"] == "10086"
        assert state.active_task.slots["refund_reason"] == "商品破损"

    def test_noop_when_no_active_task(self, processor, state, flows):
        processor.run(
            [SetSlotsCommand(command="set_slots", slots={"order_number": "10086"})],
            state,
            flows,
        )
        assert state.active_task is None

    def test_does_not_affect_paused_tasks(self, processor, state, flows):
        state.paused_tasks.append(
            TaskContext(flow_id="refund_request", step_id="listen")
        )
        processor.run(
            [SetSlotsCommand(command="set_slots", slots={"order_number": "10086"})],
            state,
            flows,
        )
        # Paused task unchanged
        assert state.paused_tasks[0].slots == {}


# ---------------------------------------------------------------------------
# cancel_flow
# ---------------------------------------------------------------------------


class TestCancelFlow:
    def test_cancels_active_task(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [CancelFlowCommand(command="cancel_flow")],
            state,
            flows,
        )
        assert state.active_task is None

        sf = state.active_system_flow
        assert isinstance(sf, CanceledSystemContext)
        assert sf.canceled_flow_id == "refund_request"
        assert sf.canceled_flow_name == "退款申请"

    def test_noop_when_no_active_task(self, processor, state, flows):
        processor.run(
            [CancelFlowCommand(command="cancel_flow")],
            state,
            flows,
        )
        assert state.active_task is None
        assert state.active_system_flow is None


# ---------------------------------------------------------------------------
# resume_flow
# ---------------------------------------------------------------------------


class TestResumeFlow:
    def test_resumes_paused_task(self, processor, state, flows):
        state.paused_tasks.append(
            TaskContext(flow_id="refund_request", step_id="ask_order_number")
        )
        processor.run(
            [ResumeFlowCommand(command="resume_flow", flow="refund_request")],
            state,
            flows,
        )
        assert state.active_task is not None
        assert state.active_task.flow_id == "refund_request"
        assert state.active_task.step_id == "ask_order_number"
        assert len(state.paused_tasks) == 0

        sf = state.active_system_flow
        assert isinstance(sf, ResumedSystemContext)
        assert sf.resumed_flow_id == "refund_request"

    def test_noop_when_flow_not_in_paused(self, processor, state, flows):
        processor.run(
            [ResumeFlowCommand(command="resume_flow", flow="refund_request")],
            state,
            flows,
        )
        assert state.active_task is None
        assert state.active_system_flow is None

    def test_noop_when_already_active(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [ResumeFlowCommand(command="resume_flow", flow="refund_request")],
            state,
            flows,
        )
        # Unchanged
        assert state.active_task.step_id == "listen"
        assert state.active_system_flow is None


# ---------------------------------------------------------------------------
# Command sequences
# ---------------------------------------------------------------------------


class TestCommandSequences:
    def test_cancel_then_start(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [
                CancelFlowCommand(command="cancel_flow"),
                StartFlowCommand(command="start_flow", flow="order_status_query"),
            ],
            state,
            flows,
        )
        assert state.active_task.flow_id == "order_status_query"
        assert isinstance(state.active_system_flow, StartedSystemContext)

    def test_set_slots_then_start_different_flow(self, processor, state, flows):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        processor.run(
            [
                SetSlotsCommand(
                    command="set_slots", slots={"order_number": "10086"}
                ),
                StartFlowCommand(command="start_flow", flow="logistics_tracking"),
            ],
            state,
            flows,
        )
        # Old task paused with its slots preserved
        assert len(state.paused_tasks) == 1
        assert state.paused_tasks[0].flow_id == "refund_request"
        assert state.paused_tasks[0].slots["order_number"] == "10086"
        # New task active
        assert state.active_task.flow_id == "logistics_tracking"

    def test_resume_with_set_slots(self, processor, state, flows):
        state.paused_tasks.append(
            TaskContext(flow_id="refund_request", step_id="listen")
        )
        processor.run(
            [
                ResumeFlowCommand(command="resume_flow", flow="refund_request"),
                SetSlotsCommand(
                    command="set_slots", slots={"refund_reason": "不想要了"}
                ),
            ],
            state,
            flows,
        )
        assert state.active_task.flow_id == "refund_request"
        assert state.active_task.slots["refund_reason"] == "不想要了"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_commands_is_noop(self, processor, state, flows):
        processor.run([], state, flows)
        assert state.active_task is None
        assert state.active_system_flow is None

    def test_resume_clears_existing_system_flow(self, processor, state, flows):
        state.paused_tasks.append(
            TaskContext(flow_id="refund_request", step_id="listen")
        )
        state.active_system_flow = StartedSystemContext(
            started_flow_id="old", started_flow_name="Old"
        )
        processor.run(
            [ResumeFlowCommand(command="resume_flow", flow="refund_request")],
            state,
            flows,
        )
        assert isinstance(state.active_system_flow, ResumedSystemContext)

    def test_cancel_with_paused_tasks_does_not_touch_them(
        self, processor, state, flows
    ):
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        state.paused_tasks.append(
            TaskContext(flow_id="order_status_query", step_id="start")
        )
        processor.run(
            [CancelFlowCommand(command="cancel_flow")],
            state,
            flows,
        )
        assert state.active_task is None
        assert len(state.paused_tasks) == 1
        assert state.paused_tasks[0].flow_id == "order_status_query"


# ---------------------------------------------------------------------------
# State helper methods
# ---------------------------------------------------------------------------


class TestDialogueStateTaskMethods:
    def test_start_new_task(self):
        state = DialogueState()
        task = state.start_new_task("refund_request", "start")
        assert task.flow_id == "refund_request"
        assert task.step_id == "start"
        assert state.active_task is task

    def test_start_new_task_overwrites_previous(self):
        state = DialogueState()
        state.active_task = TaskContext(flow_id="old", step_id="x")
        state.start_new_task("new", "start")
        assert state.active_task.flow_id == "new"

    def test_pause_active_task(self):
        state = DialogueState()
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        paused = state.pause_active_task()
        assert paused.flow_id == "refund_request"
        assert state.active_task is None
        assert len(state.paused_tasks) == 1

    def test_pause_active_task_when_none(self):
        state = DialogueState()
        assert state.pause_active_task() is None

    def test_cancel_active_task(self):
        state = DialogueState()
        state.active_task = TaskContext(flow_id="refund_request", step_id="listen")
        canceled = state.cancel_active_task()
        assert canceled.flow_id == "refund_request"
        assert state.active_task is None

    def test_cancel_active_task_when_none(self):
        state = DialogueState()
        assert state.cancel_active_task() is None

    def test_resume_task_found(self):
        state = DialogueState()
        state.paused_tasks = [
            TaskContext(flow_id="a", step_id="s1"),
            TaskContext(flow_id="b", step_id="s2"),
        ]
        resumed = state.resume_task("b")
        assert resumed.flow_id == "b"
        assert resumed.step_id == "s2"
        assert [t.flow_id for t in state.paused_tasks] == ["a"]
        assert state.active_task is resumed

    def test_resume_task_not_found(self):
        state = DialogueState()
        state.paused_tasks = [TaskContext(flow_id="a", step_id="s1")]
        assert state.resume_task("b") is None
        assert len(state.paused_tasks) == 1

    def test_set_slot(self):
        state = DialogueState()
        state.active_task = TaskContext(flow_id="f", step_id="s")
        state.set_slot("key", "value")
        assert state.active_task.slots["key"] == "value"

    def test_set_slot_no_active_task(self):
        state = DialogueState()
        state.set_slot("key", "value")  # does not raise
        assert state.active_task is None

    def test_activate_system_flow(self):
        state = DialogueState()
        ctx = StartedSystemContext(
            started_flow_id="f", started_flow_name="n"
        )
        state.activate_system_flow(ctx)
        assert state.active_system_flow is ctx