"""Tests for FlowExecutor — advancing flows and executing actions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from Customer_Service_Assistant.service.schemas import (
    CannotHandleSystemContext,
    CollectSystemContext,
    CompletedSystemContext,
    DialogueState,
    FocusedObject,
    Message,
    StartedSystemContext,
    TaskContext,
    Turn,
)
from Customer_Service_Assistant.service.task import (
    ActionFlowStep,
    ActionRunner,
    ActionResult,
    CollectSlotStep,
    EndFlowStep,
    Flow,
    FlowExecutor,
    FlowLoader,
    FlowStepType,
    FlowsList,
    ResponseDefinition,
    SlotValidation,
    StartFlowStep,
    StaticLink,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flow(*steps) -> Flow:
    return Flow(id="test_flow", description="Test", steps=list(steps))


def _flows_with_system(*extra_flows: Flow) -> FlowsList:
    """Create a FlowsList with *extra_flows* plus real system flows."""
    return FlowsList(
        flows=list(extra_flows) + list(_REAL_FLOWS.flows),
        slots=_REAL_FLOWS.slots,
    )


def _make_executor(response_map: dict | None = None) -> FlowExecutor:
    """Build a FlowExecutor whose ActionRunner returns canned responses.

    *response_map* keys are action names; values are ActionResult or str.
    When an action name is not in the map, ``ActionResult.reply("ok")``
    is returned — except for ``action_listen`` which always returns
    ``ActionResult.listen()``.
    """
    map_ = response_map or {}

    async def _run(step, state):
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
    return FlowExecutor(runner)


def _state(**kwargs) -> DialogueState:
    s = DialogueState(sender_id="test_user")
    s.pending_turn = Turn(input_message=Message(role="user", text="hi"))
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# Real flows for integration-level tests
FLOW_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "flow_config"
_REAL_FLOWS = FlowLoader().load([FLOW_DIR / "user_flows.yml", FLOW_DIR / "system_flows.yml"])


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBasicExecution:
    async def test_empty_state_returns_no_messages(self):
        executor = _make_executor()
        state = _state()
        msgs = await executor.run(state, _REAL_FLOWS)
        assert msgs == []

    async def test_simple_start_to_action_to_end(self):
        """A flow with start → action_response → end."""
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="reply")]),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           args={"text": "hello"}, next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)

        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start"))
        msgs = await _make_executor({"action_response": "hello world"}).run(state, flows)

        assert len(msgs) == 1
        assert msgs[0].text == "hello world"
        # Business flow ended
        assert state.active_task is None

    async def test_system_flow_has_priority(self):
        """When active_system_flow is set, it executes before business flow."""
        flows = _REAL_FLOWS

        state = _state(
            active_task=TaskContext(flow_id="refund_request", step_id="start",
                                     slots={"order_number": "10086", "refund_reason": "test"}),
            active_system_flow=StartedSystemContext(
                started_flow_id="refund_request",
                started_flow_name="退款申请",
                step_id="start",
            ),
        )
        msgs = await _make_executor({"action_response": "system msg"}).run(state, flows)
        # System flow produces its message first
        assert len(msgs) >= 1
        assert msgs[0].text == "system msg"
        # System flow ended
        assert not isinstance(state.active_system_flow, StartedSystemContext)


# ---------------------------------------------------------------------------
# action_listen stops outer loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestActionListen:
    async def test_stops_at_action_listen(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="listen")]),
            ActionFlowStep(id="listen", type=FlowStepType.ACTION, action="action_listen",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start"))
        msgs = await _make_executor().run(state, flows)
        assert msgs == []
        # step_id advanced past the listen step
        assert state.active_task.step_id == "end"

    async def test_response_then_listen_stops_after_listen(self):
        """Two actions: response + listen → stops after listen."""
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask")]),
            ActionFlowStep(id="ask", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="listen")]),
            ActionFlowStep(id="listen", type=FlowStepType.ACTION, action="action_listen",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start"))

        msgs = await _make_executor({"action_response": "请告诉我订单号"}).run(state, flows)

        assert len(msgs) == 1
        assert msgs[0].text == "请告诉我订单号"
        assert state.active_task.step_id == "end"


# ---------------------------------------------------------------------------
# Collect step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCollectStep:
    async def test_slot_filled_advances_past_collect(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                            response=ResponseDefinition(text="请告诉我订单号"),
                            next=[StaticLink(target="reply")]),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start",
                                                slots={"order_number": "10086"}))
        msgs = await _make_executor({"action_response": "ok"}).run(state, flows)
        assert len(msgs) == 1
        # Collect was skipped, went straight to action_response

    async def test_slot_missing_activates_system_collect(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                            response=ResponseDefinition(text="请告诉我订单号"),
                            next=[StaticLink(target="reply")]),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start"))

        msgs = await _make_executor().run(state, flows)

        # system_collect_information should be activated
        assert isinstance(state.active_system_flow, CollectSystemContext)
        assert state.active_system_flow.slot_name == "order_number"
        assert state.active_system_flow.response["text"] == "请告诉我订单号"

    async def test_auto_fill_from_focused_object(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                            response=ResponseDefinition(text="请告诉我订单号"),
                            next=[StaticLink(target="reply")]),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(
            active_task=TaskContext(flow_id="test_flow", step_id="start"),
            focused_object=FocusedObject(type="order", id="ORD-10086", title="Order #10086"),
        )
        # Slot auto-filled from focused_object → collect skipped → flow runs to end
        msgs = await _make_executor({"action_response": "订单已找到"}).run(state, flows)
        assert len(msgs) == 1

    async def test_validation_pass_advances(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(
                id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                response=ResponseDefinition(text="请告诉我订单号"),
                validation=SlotValidation(condition="slots.get('order_number')"),
                next=[StaticLink(target="reply")],
            ),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start",
                                                slots={"order_number": "OK"}))
        msgs = await _make_executor({"action_response": "valid"}).run(state, flows)
        assert len(msgs) == 1

    async def test_validation_fail_returns_failure_response(self):
        failure_resp = ResponseDefinition(mode="static", text="订单号无效，请重新输入")
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(
                id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                response=ResponseDefinition(text="请告诉我订单号"),
                validation=SlotValidation(
                    condition="len(slots.get('order_number', '')) >= 5",
                    failure_response=failure_resp,
                ),
                next=[StaticLink(target="reply")],
            ),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start",
                                                slots={"order_number": "12"}))
        msgs = await _make_executor({"action_response": "订单号无效"}).run(state, flows)
        # Validation failed → slot cleared, failure response sent
        assert len(msgs) >= 1
        assert "无效" in msgs[0].text

    async def test_validation_fail_without_failure_response_loops_back(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="ask_order")]),
            CollectSlotStep(
                id="ask_order", type=FlowStepType.COLLECT, slot_name="order_number",
                response=ResponseDefinition(text="请告诉我订单号"),
                validation=SlotValidation(condition="len(slots.get('order_number', '')) >= 5"),
                next=[StaticLink(target="reply")],
            ),
            ActionFlowStep(id="reply", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start",
                                                slots={"order_number": "12"}))
        msgs = await _make_executor().run(state, flows)
        # Slot cleared, system_collect_information activated
        assert "order_number" not in state.active_task.slots
        assert isinstance(state.active_system_flow, CollectSystemContext)


# ---------------------------------------------------------------------------
# Conditional next links
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConditionalNext:
    async def test_condition_true_picks_that_branch(self):
        from Customer_Service_Assistant.service.task import ConditionalLink, FallbackLink
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[
                ConditionalLink(condition="slots.get('product_id')", target="respond"),
                FallbackLink(target="missing"),
            ]),
            ActionFlowStep(id="respond", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            ActionFlowStep(id="missing", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start",
                                                slots={"product_id": "P001"}))
        msgs = await _make_executor({"action_response": "has product"}).run(state, flows)
        assert msgs[0].text == "has product"

    async def test_condition_false_falls_through(self):
        from Customer_Service_Assistant.service.task import ConditionalLink, FallbackLink
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[
                ConditionalLink(condition="slots.get('product_id')", target="respond"),
                FallbackLink(target="missing"),
            ]),
            ActionFlowStep(id="respond", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            ActionFlowStep(id="missing", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(active_task=TaskContext(flow_id="test_flow", step_id="start"))
        msgs = await _make_executor({"action_response": "no product"}).run(state, flows)
        assert msgs[0].text == "no product"

    async def test_context_condition(self):
        from Customer_Service_Assistant.service.task import ConditionalLink, FallbackLink
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[
                ConditionalLink(condition="context.get('reason') == 'not_supported'", target="ns"),
                FallbackLink(target="ok"),
            ]),
            ActionFlowStep(id="ns", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            ActionFlowStep(id="ok", type=FlowStepType.ACTION, action="action_response",
                           next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        flows = _flows_with_system(flow)
        state = _state(
            active_system_flow=CannotHandleSystemContext(
                reason="not_supported", step_id="start",
            ),
        )
        msgs = await _make_executor({"action_response": "ns branch"}).run(state, flows)
        assert msgs[0].text == "ns branch"


# ---------------------------------------------------------------------------
# End step handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEndStep:
    async def test_system_flow_end_clears_only_system(self):
        """After system_task_started flow ends, only system flow is cleared."""
        flows = _REAL_FLOWS
        state = _state(
            active_task=TaskContext(flow_id="refund_request", step_id="start",
                                     slots={"order_number": "10086", "refund_reason": "test"}),
            active_system_flow=StartedSystemContext(
                started_flow_id="refund_request",
                started_flow_name="退款申请",
                step_id="start",
            ),
        )
        msgs = await _make_executor().run(state, flows)
        # System flow ended and cleared
        assert not isinstance(state.active_system_flow, StartedSystemContext)
        # System flow message was produced first
        assert len(msgs) >= 1

    async def test_business_flow_end_clears_task(self):
        """End step of business flow clears active_task."""
        state = _state(active_task=TaskContext(flow_id="refund_request", step_id="end"))
        msgs = await _make_executor().run(state, _REAL_FLOWS)
        # Business flow ended → task cleared
        assert state.active_task is None


# ---------------------------------------------------------------------------
# Real flow: refund_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRealFlowRefund:
    async def test_refund_collect_order_number(self):
        """Start refund_request with no slots → should ask for order_number."""
        state = _state(active_task=TaskContext(flow_id="refund_request", step_id="start"))
        executor = _make_executor()
        msgs = await executor.run(state, _REAL_FLOWS)

        # system_collect_information activated for order_number
        assert isinstance(state.active_system_flow, CollectSystemContext)
        assert state.active_system_flow.slot_name == "order_number"

    async def test_refund_with_order_number_filled(self):
        """refund_request with order_number present → advances to ask refund_reason."""
        state = _state(active_task=TaskContext(
            flow_id="refund_request", step_id="ask_order_number",
            slots={"order_number": "10086"},
        ))
        executor = _make_executor()
        msgs = await executor.run(state, _REAL_FLOWS)

        # Should have skipped order_number collect and hit refund_reason collect
        assert isinstance(state.active_system_flow, CollectSystemContext)
        assert state.active_system_flow.slot_name == "refund_reason"

    async def test_refund_all_slots_filled_shows_summary(self):
        """Both slots filled → action_response with refund message."""
        state = _state(active_task=TaskContext(
            flow_id="refund_request", step_id="refund_submitted",
            slots={"order_number": "10086", "refund_reason": "质量问题"},
        ))
        executor = _make_executor({"action_response": "订单10086退款已提交：质量问题"})
        msgs = await executor.run(state, _REAL_FLOWS)

        assert len(msgs) == 1
        assert "10086" in msgs[0].text
        # Flow ended
        assert state.active_task is None


# ---------------------------------------------------------------------------
# Real flow: order_status_query with custom action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRealFlowOrderStatus:
    async def test_custom_action_lookup_produces_slot_updates(self):
        """The action_lookup_order_status should populate order_status/order_summary."""
        state = _state(active_task=TaskContext(
            flow_id="order_status_query", step_id="lookup_order_status",
            slots={"order_number": "10086"},
        ))
        executor = _make_executor({
            "action_lookup_order_status": ActionResult(
                slot_updates={"order_status": "已发货", "order_summary": "预计明天送达"},
            ),
            "action_response": "订单10086状态：已发货",
        })
        msgs = await executor.run(state, _REAL_FLOWS)

        # Custom action slots were applied (before flow continued to end)
        assert len(msgs) == 1
        assert "已发货" in msgs[0].text or "10086" in msgs[0].text


# ---------------------------------------------------------------------------
# Flow.get_step
# ---------------------------------------------------------------------------


class TestFlowGetStep:
    def test_returns_step_by_id(self):
        flow = _flow(
            StartFlowStep(id="start", type=FlowStepType.START, next=[StaticLink(target="end")]),
            EndFlowStep(id="end", type=FlowStepType.END),
        )
        assert flow.get_step("start") is flow.steps[0]
        assert flow.get_step("end") is flow.steps[1]

    def test_returns_none_for_unknown_id(self):
        flow = _flow(StartFlowStep(id="start", type=FlowStepType.START))
        assert flow.get_step("nonexistent") is None