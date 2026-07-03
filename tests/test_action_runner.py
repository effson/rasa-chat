"""Tests for ActionRunner, ActionResult, render_template, and ActionRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from Customer_Service_Assistant.service.schemas import (
    CollectSystemContext,
    DialogueState,
    Message,
    StartedSystemContext,
    TaskContext,
    Turn,
)
from Customer_Service_Assistant.service.task import (
    ActionFlowStep,
    ActionRegistry,
    ActionResult,
    ActionRunner,
    create_default_registry,
    FlowStepType,
    render_template,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def runner(mock_llm):
    return ActionRunner(llm=mock_llm)


@pytest.fixture
def state():
    s = DialogueState(sender_id="test_user")
    s.active_task = TaskContext(flow_id="refund_request", step_id="listen")
    s.pending_turn = Turn(
        input_message=Message(role="user", text="订单号是10086")
    )
    return s


def _static_step(text: str) -> ActionFlowStep:
    return ActionFlowStep(
        id="respond",
        type=FlowStepType.ACTION,
        action="action_response",
        args={"mode": "static", "text": text},
    )


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


class TestActionResult:
    def test_listen_factory(self):
        r = ActionResult.listen()
        assert r.should_listen is True
        assert r.messages == []
        assert r.slot_updates == {}

    def test_reply_factory(self):
        r = ActionResult.reply("你好")
        assert r.should_listen is False
        assert len(r.messages) == 1
        assert r.messages[0].text == "你好"
        assert r.messages[0].role == "bot"

    def test_defaults(self):
        r = ActionResult()
        assert r.messages == []
        assert r.slot_updates == {}
        assert r.should_listen is False

    def test_with_slot_updates(self):
        r = ActionResult(slot_updates={"order_status": "已发货"})
        assert r.slot_updates == {"order_status": "已发货"}


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_slots_substitution(self):
        result = render_template(
            "订单{{ slots.order_number }}已提交", slots={"order_number": "10086"}
        )
        assert result == "订单10086已提交"

    def test_context_substitution(self):
        ctx = StartedSystemContext(
            started_flow_id="refund_request",
            started_flow_name="退款申请",
        )
        result = render_template(
            "好的，我们先处理{{ context.started_flow_name }}。", context=ctx
        )
        assert result == "好的，我们先处理退款申请。"

    def test_bare_key_substitution(self):
        result = render_template(
            "用户：{{ user_message }}",
            extra={"user_message": "你好"},
        )
        assert result == "用户：你好"

    def test_multiple_variables(self):
        result = render_template(
            "订单{{ slots.order_number }}：{{ slots.order_status }}",
            slots={"order_number": "10086", "order_status": "已发货"},
        )
        assert result == "订单10086：已发货"

    def test_missing_slot_keeps_placeholder(self):
        result = render_template(
            "订单{{ slots.order_number }}", slots={}
        )
        assert result == "订单{{ slots.order_number }}"

    def test_missing_context_keeps_placeholder(self):
        result = render_template(
            "{{ context.nonexistent }}", context=None
        )
        assert result == "{{ context.nonexistent }}"

    def test_missing_extra_keeps_placeholder(self):
        result = render_template("{{ missing }}", extra={})
        assert result == "{{ missing }}"

    def test_no_placeholders_returns_original(self):
        result = render_template("普通文本，无变量")
        assert result == "普通文本，无变量"

    def test_empty_string(self):
        assert render_template("") == ""

    def test_text_with_int_slot_value(self):
        result = render_template(
            "数量：{{ slots.count }}", slots={"count": 5}
        )
        assert result == "数量：5"


# ---------------------------------------------------------------------------
# action_listen
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestActionListen:
    async def test_returns_listen_result(self, runner, state):
        step = ActionFlowStep(
            id="listen",
            type=FlowStepType.ACTION,
            action="action_listen",
        )
        result = await runner.run(step, state)
        assert result.should_listen is True
        assert result.messages == []


# ---------------------------------------------------------------------------
# action_response — static mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestActionResponseStatic:
    async def test_returns_rendered_text(self, runner, state):
        state.active_task.slots["order_number"] = "10086"
        step = _static_step("订单号是{{ slots.order_number }}")
        result = await runner.run(step, state)
        assert result.should_listen is False
        assert result.messages[0].text == "订单号是10086"

    async def test_default_mode_is_static(self, runner, state):
        step = ActionFlowStep(
            id="respond",
            type=FlowStepType.ACTION,
            action="action_response",
            args={"text": "你好，有什么可以帮你？"},
        )
        result = await runner.run(step, state)
        assert result.messages[0].text == "你好，有什么可以帮你？"

    async def test_context_variables_in_static(self, runner, state):
        state.active_system_flow = StartedSystemContext(
            started_flow_id="refund_request",
            started_flow_name="退款申请",
        )
        step = _static_step("好的，我们先处理{{ context.started_flow_name }}。")
        result = await runner.run(step, state)
        assert "退款申请" in result.messages[0].text

    async def test_no_slots_no_context_renders_plain(self, runner):
        s = DialogueState(sender_id="t")
        s.pending_turn = Turn(input_message=Message(role="user", text="hi"))
        step = _static_step("你好")
        result = await runner.run(step, s)
        assert result.messages[0].text == "你好"


# ---------------------------------------------------------------------------
# action_response — rephrase mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestActionResponseRephrase:
    async def test_calls_llm_with_rendered_prompt(self, runner, mock_llm, state):
        mock_llm.ainvoke.return_value = MagicMock(content="  这是改写后的回复  ")
        state.active_task.slots["order_number"] = "10086"

        step = ActionFlowStep(
            id="not_supported",
            type=FlowStepType.ACTION,
            action="action_response",
            args={
                "mode": "rephrase",
                "text": "建议回复：订单{{ slots.order_number }}",
                "prompt": "改写以下内容：{{ current_response }}",
            },
        )
        result = await runner.run(step, state)

        # LLM was called
        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "订单10086" in prompt_text  # current_response rendered
        assert result.messages[0].text == "这是改写后的回复"

    async def test_falls_back_to_current_response_on_empty_llm(self, runner, mock_llm, state):
        mock_llm.ainvoke.return_value = MagicMock(content="")

        step = ActionFlowStep(
            id="s",
            type=FlowStepType.ACTION,
            action="action_response",
            args={
                "mode": "rephrase",
                "text": "建议回复",
                "prompt": "改写",
            },
        )
        result = await runner.run(step, state)
        assert result.messages[0].text == "建议回复"


# ---------------------------------------------------------------------------
# action_response — generate mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestActionResponseGenerate:
    async def test_calls_llm_with_prompt(self, runner, mock_llm, state):
        mock_llm.ainvoke.return_value = MagicMock(content="生成的内容")

        step = ActionFlowStep(
            id="generate_reply",
            type=FlowStepType.ACTION,
            action="action_response",
            args={
                "mode": "generate",
                "prompt": "生成一句回复：{{ user_message }}",
            },
        )
        result = await runner.run(step, state)

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "订单号是10086" in prompt_text  # user_message rendered
        assert result.messages[0].text == "生成的内容"

    async def test_returns_empty_on_empty_llm_response(self, runner, mock_llm, state):
        mock_llm.ainvoke.return_value = MagicMock(content="")

        step = ActionFlowStep(
            id="g",
            type=FlowStepType.ACTION,
            action="action_response",
            args={"mode": "generate", "prompt": "生成"},
        )
        result = await runner.run(step, state)
        assert result.messages[0].text == ""


# ---------------------------------------------------------------------------
# args resolution — context.response string reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestArgsResolution:
    async def test_resolves_context_dot_response_string(self, runner, state):
        state.active_system_flow = CollectSystemContext(
            slot_name="order_number",
            response={"mode": "static", "text": "请告诉我你的订单号。"},
        )
        step = ActionFlowStep(
            id="ask",
            type=FlowStepType.ACTION,
            action="action_response",
            args="context.response",
        )
        result = await runner.run(step, state)
        assert result.messages[0].text == "请告诉我你的订单号。"

    async def test_invalid_string_ref_returns_empty(self, runner, state):
        step = ActionFlowStep(
            id="ask",
            type=FlowStepType.ACTION,
            action="action_response",
            args="context.nonexistent",
        )
        result = await runner.run(step, state)
        # args resolved to {} — no text, fallback to static rendering of ""
        assert result.messages[0].text == ""


# ---------------------------------------------------------------------------
# Custom actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCustomActions:
    async def test_dispatches_to_registered_handler(self, mock_llm, state):
        registry = ActionRegistry()
        registry.register(
            "action_lookup_order_status",
            AsyncMock(
                return_value=ActionResult(
                    slot_updates={"order_status": "已发货", "order_summary": "预计明天送达"},
                )
            ),
        )
        runner = ActionRunner(llm=mock_llm, registry=registry)

        step = ActionFlowStep(
            id="lookup",
            type=FlowStepType.ACTION,
            action="action_lookup_order_status",
        )
        result = await runner.run(step, state)

        assert result.slot_updates["order_status"] == "已发货"
        assert result.slot_updates["order_summary"] == "预计明天送达"

    async def test_unknown_action_returns_empty(self, runner, state):
        step = ActionFlowStep(
            id="unknown",
            type=FlowStepType.ACTION,
            action="nonexistent_action",
        )
        result = await runner.run(step, state)
        assert result.messages == []
        assert result.slot_updates == {}
        assert result.should_listen is False

    async def test_custom_action_receives_args_and_state(self, mock_llm, state):
        received_args = None
        received_state = None

        async def handler(args, st):
            nonlocal received_args, received_state
            received_args = args
            received_state = st
            return ActionResult(messages=[Message(role="bot", text="done")])

        registry = ActionRegistry()
        registry.register("my_action", handler)
        runner = ActionRunner(llm=mock_llm, registry=registry)

        step = ActionFlowStep(
            id="custom",
            type=FlowStepType.ACTION,
            action="my_action",
            args={"key": "value"},
        )
        await runner.run(step, state)

        assert received_args == {"key": "value"}
        assert received_state is state


# ---------------------------------------------------------------------------
# ActionRegistry
# ---------------------------------------------------------------------------


class TestActionRegistry:
    def test_register_and_get(self):
        reg = ActionRegistry()
        handler = AsyncMock()
        reg.register("test_action", handler)
        assert reg.get("test_action") is handler

    def test_get_unknown_returns_none(self):
        reg = ActionRegistry()
        assert reg.get("unknown") is None

    def test_overwrite_handler(self):
        reg = ActionRegistry()
        h1 = AsyncMock()
        h2 = AsyncMock()
        reg.register("a", h1)
        reg.register("a", h2)
        assert reg.get("a") is h2


class TestDefaultRegistry:
    def test_all_three_custom_actions_registered(self):
        reg = create_default_registry()
        assert reg.get("action_lookup_order_status") is not None
        assert reg.get("action_lookup_logistics") is not None
        assert reg.get("action_recommend_similar_products") is not None

    @pytest.mark.asyncio
    async def test_lookup_order_status_returns_slot_updates(self):
        reg = create_default_registry()
        handler = reg.get("action_lookup_order_status")
        state = DialogueState(sender_id="t")
        state.active_task = TaskContext(flow_id="order_status_query", step_id="lookup")
        state.active_task.slots["order_number"] = "10086"

        result = await handler({}, state)
        assert result.slot_updates["order_status"] == "已发货"
        assert "10086" in str(result.slot_updates["order_summary"])

    @pytest.mark.asyncio
    async def test_lookup_logistics_returns_slot_updates(self):
        reg = create_default_registry()
        handler = reg.get("action_lookup_logistics")
        state = DialogueState(sender_id="t")
        state.active_task = TaskContext(flow_id="logistics_tracking", step_id="lookup")
        state.active_task.slots["order_number"] = "10086"

        result = await handler({}, state)
        assert result.slot_updates["logistics_company"] == "顺丰速运"
        assert "SF10086" in str(result.slot_updates["tracking_number"])

    @pytest.mark.asyncio
    async def test_recommend_returns_reply_message(self):
        reg = create_default_registry()
        handler = reg.get("action_recommend_similar_products")
        state = DialogueState(sender_id="t")
        state.active_task = TaskContext(flow_id="similar_product_recommendation", step_id="respond")
        state.active_task.slots["product_id"] = "P001"

        result = await handler({}, state)
        assert len(result.messages) == 1
        assert "P001" in result.messages[0].text


# ---------------------------------------------------------------------------
# Integration: render_template with system flow context
# ---------------------------------------------------------------------------


class TestTemplateRenderingIntegration:
    def test_slots_and_context_together(self):
        ctx = StartedSystemContext(
            started_flow_id="refund_request",
            started_flow_name="退款申请",
        )
        result = render_template(
            "处理{{ context.started_flow_name }}，订单{{ slots.order_number }}",
            slots={"order_number": "10086"},
            context=ctx,
        )
        assert result == "处理退款申请，订单10086"

    def test_extra_overrides_nothing(self):
        """extra keys don't shadow slots or context namespaces."""
        result = render_template(
            "{{ slots.order_number }} {{ extra.order_number }}",
            slots={"order_number": "10086"},
            extra={"order_number": "should_not_appear"},
        )
        # extra.order_number won't match because regex sees "extra" as namespace
        # but we only handle "slots" and "context" namespaces
        assert result == "10086 {{ extra.order_number }}"