"""Tests for TaskHandler — the top-level orchestrator.

Tests cover:
- Creating a TaskHandler and running it with various command sequences
- Integration with CommandProcessor (state changes) and FlowExecutor (messages)
- Error handling (malformed commands)
- End-to-end: start a flow → collect slots → reply
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    Message,
    TaskContext,
    Turn,
)
from Customer_Service_Assistant.service.task import (
    ActionRunner,
    ActionResult,
    CommandParser,
    CommandProcessor,
    FlowExecutor,
    FlowLoader,
    TaskHandler,
    TaskHandlerError,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FLOW_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "flow_config"


@pytest.fixture(scope="module")
def real_flows():
    """Load the real flow config for integration-level tests."""
    return FlowLoader().load([
        FLOW_DIR / "user_flows.yml",
        FLOW_DIR / "system_flows.yml",
    ])


@pytest.fixture
def state():
    s = DialogueState(sender_id="test_user")
    s.pending_turn = Turn(input_message=Message(role="user", text="我要退款"))
    return s


def _make_handler(flows, response_map=None):
    """Build a TaskHandler with real CommandParser/CommandProcessor and a
    mocked FlowExecutor whose ActionRunner returns canned responses."""
    map_ = response_map or {}

    async def _run(step, st):
        if step.action == "action_listen":
            return ActionResult.listen()
        val = map_.get(step.action, None)
        if val is None:
            return ActionResult.reply("ok")
        if isinstance(val, str):
            return ActionResult.reply(val)
        if isinstance(val, ActionResult):
            return val
        return ActionResult.reply("ok")

    runner = MagicMock(spec=ActionRunner)
    runner.run = AsyncMock(side_effect=_run)

    return TaskHandler(
        command_processor=CommandProcessor(),
        command_parser=CommandParser(),
        flow_executor=FlowExecutor(runner),
        flows=flows,
    )


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBasicExecution:
    async def test_empty_commands_returns_empty(self, real_flows, state):
        handler = _make_handler(real_flows)
        msgs = await handler.run([], state)
        assert msgs == []

    async def test_start_flow_produces_system_started_message(self, real_flows, state):
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，我们先处理退款申请。"},
        )
        msgs = await handler.run(
            [{"command": "start_flow", "flow": "refund_request"}],
            state,
        )
        # system_task_started should produce a message
        assert len(msgs) >= 1
        assert "退款申请" in msgs[0].text
        # active_task should be set
        assert state.active_task is not None
        assert state.active_task.flow_id == "refund_request"

    async def test_set_slots_on_active_task(self, real_flows, state):
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
        )
        handler = _make_handler(real_flows)
        await handler.run(
            [{"command": "set_slots", "slots": {"order_number": "10086"}}],
            state,
        )
        assert state.active_task.slots["order_number"] == "10086"

    async def test_cancel_flow_produces_message(self, real_flows, state):
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，退款申请先帮你取消。"},
        )
        msgs = await handler.run(
            [{"command": "cancel_flow"}],
            state,
        )
        assert state.active_task is None
        assert len(msgs) >= 1
        assert "取消" in msgs[0].text

    async def test_resume_flow_produces_message(self, real_flows, state):
        state.paused_tasks.append(
            TaskContext(flow_id="refund_request", step_id="ask_refund_reason",
                        slots={"order_number": "10086"}),
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，我们继续刚才的退款申请。"},
        )
        msgs = await handler.run(
            [{"command": "resume_flow", "flow": "refund_request"}],
            state,
        )
        assert state.active_task is not None
        assert state.active_task.flow_id == "refund_request"
        assert len(msgs) >= 1
        assert "继续" in msgs[0].text


# ---------------------------------------------------------------------------
# Multi-command sequences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCommandSequences:
    async def test_cancel_then_start_new_flow(self, real_flows, state):
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，我们先处理物流查询。"},
        )
        msgs = await handler.run(
            [
                {"command": "cancel_flow"},
                {"command": "start_flow", "flow": "logistics_tracking"},
            ],
            state,
        )
        assert state.active_task.flow_id == "logistics_tracking"
        assert len(msgs) >= 1

    async def test_set_slots_then_start(self, real_flows, state):
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="listen",
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，我们先处理物流查询。"},
        )
        await handler.run(
            [
                {"command": "set_slots", "slots": {"order_number": "10086"}},
                {"command": "start_flow", "flow": "logistics_tracking"},
            ],
            state,
        )
        # Old task paused with slots preserved
        assert len(state.paused_tasks) == 1
        assert state.paused_tasks[0].flow_id == "refund_request"
        assert state.paused_tasks[0].slots["order_number"] == "10086"
        # New task active
        assert state.active_task.flow_id == "logistics_tracking"


# ---------------------------------------------------------------------------
# End-to-end: refund flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEndToEndRefund:
    async def test_start_refund_collects_order_number(self, real_flows, state):
        """Start refund_request → system_task_started message + collect order_number."""
        handler = _make_handler(
            real_flows,
            {"action_response": "请告诉我你的订单号。"},
        )
        msgs = await handler.run(
            [{"command": "start_flow", "flow": "refund_request"}],
            state,
        )

        # Should have at least 2 messages: "started" + "ask order_number"
        all_text = " ".join(m.text for m in msgs if m.text)
        assert "退款申请" in all_text or "订单号" in all_text
        assert state.active_task.flow_id == "refund_request"

    async def test_refund_flow_with_all_slots(self, real_flows, state):
        """When all slots are pre-filled, refund produces confirmation directly."""
        state.active_task = TaskContext(
            flow_id="refund_request",
            step_id="refund_submitted",
            slots={"order_number": "10086", "refund_reason": "质量问题"},
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，订单10086的退款申请已提交，原因是：质量问题。"},
        )
        msgs = await handler.run([], state)

        assert len(msgs) == 1
        assert "10086" in msgs[0].text
        # Task completed
        assert state.active_task is None

    async def test_refund_then_switch_to_logistics(self, real_flows, state):
        """Start refund, then switch to logistics → interrupted message."""
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
        )
        handler = _make_handler(
            real_flows,
            {"action_response": "请告诉我你的订单号。"},
        )
        msgs = await handler.run(
            [{"command": "start_flow", "flow": "logistics_tracking"}],
            state,
        )
        # Old task paused
        assert len(state.paused_tasks) == 1
        assert state.paused_tasks[0].flow_id == "refund_request"
        # New task active
        assert state.active_task.flow_id == "logistics_tracking"
        # Interrupted message produced
        all_text = " ".join(m.text for m in msgs if m.text)
        assert "退款申请" in all_text or "物流" in all_text or "放一放" in all_text or "订单号" in all_text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorHandling:
    async def test_malformed_command_skipped(self, real_flows, state):
        """Malformed commands are logged and skipped; other commands still run."""
        handler = _make_handler(
            real_flows,
            {"action_response": "好的，我们先处理退款申请。"},
        )
        msgs = await handler.run(
            [
                {"command": "unknown_command"},  # skipped
                {"command": "start_flow", "flow": "refund_request"},  # processed
            ],
            state,
        )
        # The valid command was processed
        assert state.active_task is not None
        assert state.active_task.flow_id == "refund_request"

    async def test_malformed_command_no_fields(self, real_flows, state):
        """Command dict missing 'command' field is skipped gracefully."""
        handler = _make_handler(real_flows)
        msgs = await handler.run([{"foo": "bar"}], state)
        assert msgs == []
        assert state.active_task is None

    async def test_invalid_flow_raises(self, real_flows, state):
        """start_flow targeting a nonexistent flow raises TaskHandlerError."""
        handler = _make_handler(real_flows)
        with pytest.raises(TaskHandlerError, match="命令处理失败"):
            await handler.run(
                [{"command": "start_flow", "flow": "nonexistent_flow"}],
                state,
            )

    async def test_system_flow_rejected_by_parser(self, real_flows, state):
        """start_flow targeting a system flow is rejected by CommandParser
        (skipped as malformed), so state is unchanged."""
        handler = _make_handler(real_flows)
        msgs = await handler.run(
            [{"command": "start_flow", "flow": "system_task_started"}],
            state,
        )
        # Command was skipped — no state change, no messages
        assert state.active_task is None
        assert msgs == []


# ---------------------------------------------------------------------------
# No-op scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNoOpScenarios:
    async def test_start_same_flow_twice(self, real_flows, state):
        """Starting the already-active flow is a no-op."""
        state.active_task = TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
        )
        handler = _make_handler(real_flows)
        await handler.run(
            [{"command": "start_flow", "flow": "refund_request"}],
            state,
        )
        assert state.active_task.flow_id == "refund_request"
        assert state.active_task.step_id == "ask_order_number"  # unchanged

    async def test_cancel_with_no_active_task(self, real_flows, state):
        handler = _make_handler(real_flows)
        msgs = await handler.run([{"command": "cancel_flow"}], state)
        assert state.active_task is None
        assert msgs == []

    async def test_resume_not_in_paused(self, real_flows, state):
        handler = _make_handler(real_flows)
        msgs = await handler.run(
            [{"command": "resume_flow", "flow": "refund_request"}],
            state,
        )
        assert state.active_task is None
        assert msgs == []

    async def test_set_slots_without_active_task(self, real_flows, state):
        handler = _make_handler(real_flows)
        msgs = await handler.run(
            [{"command": "set_slots", "slots": {"order_number": "10086"}}],
            state,
        )
        assert msgs == []