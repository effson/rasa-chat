"""Unit tests for TurnPlanValidator — step 6 of the DialogueEngine design."""

from __future__ import annotations

import pytest

from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    FocusedObject,
    TaskContext,
    TurnPlan,
    ValidationResult,
)
from Customer_Service_Assistant.service.validator import TurnPlanValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plan(**overrides) -> TurnPlan:
    """Build a TurnPlan with sensible defaults, overridable per test."""
    data: dict = {
        "direction": "task",
        "reason": "用户要求查询物流",
        "flow_id": "logistics_tracking",
        "action": "start",
    }
    data.update(overrides)
    return TurnPlan(**data)


def make_state(**overrides) -> DialogueState:
    """Build a DialogueState with defaults, overridable per test."""
    data: dict = {"sender_id": "u1"}
    data.update(overrides)
    return DialogueState(**data)


@pytest.fixture
def validator() -> TurnPlanValidator:
    return TurnPlanValidator()


# ===================================================================
# Valid plans
# ===================================================================


class TestValidPlans:
    def test_valid_task_start(self, validator):
        plan = make_plan(
            direction="task", flow_id="logistics_tracking", action="start",
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is True
        assert result.direction == "task"
        assert result.issues == []

    def test_valid_task_resume_from_paused(self, validator):
        state = make_state(
            paused_tasks=[TaskContext(flow_id="refund_request", step_id="confirm")],
        )
        plan = make_plan(
            direction="task", flow_id="refund_request", action="resume",
        )
        result = validator.validate(plan, state)
        assert result.is_valid is True
        assert result.direction == "task"

    def test_valid_task_cancel_active(self, validator):
        state = make_state(
            active_task=TaskContext(flow_id="order_status_query", step_id="ask_order"),
        )
        plan = make_plan(
            direction="task", flow_id="order_status_query", action="cancel",
        )
        result = validator.validate(plan, state)
        assert result.is_valid is True
        assert result.direction == "task"

    def test_valid_task_continue_active(self, validator):
        state = make_state(
            active_task=TaskContext(flow_id="logistics_tracking", step_id="lookup"),
        )
        plan = make_plan(
            direction="task", flow_id="logistics_tracking", action="continue",
        )
        result = validator.validate(plan, state)
        assert result.is_valid is True
        assert result.direction == "task"

    def test_valid_knowledge(self, validator):
        plan = make_plan(
            direction="knowledge",
            knowledge_intent="商品信息",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is True
        assert result.direction == "knowledge"

    def test_valid_knowledge_return_policy(self, validator):
        plan = make_plan(
            direction="knowledge",
            knowledge_intent="退换货政策",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is True
        assert result.direction == "knowledge"

    def test_valid_chitchat(self, validator):
        plan = make_plan(
            direction="chitchat",
            flow_id=None,
            action=None,
            reason="用户在打招呼",
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is True
        assert result.direction == "chitchat"


# ===================================================================
# Self-reported invalid
# ===================================================================


class TestSelfReportedInvalid:
    def test_direction_invalid_with_missing_info(self, validator):
        plan = make_plan(
            direction="invalid",
            reason="用户说'这个怎么办'但没有指定对象",
            missing_info="缺少对象引用",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert result.direction is None
        assert any("缺少对象引用" in i for i in result.issues)

    def test_direction_invalid_with_conflicts(self, validator):
        plan = make_plan(
            direction="invalid",
            reason="用户消息同时像查订单和问政策",
            conflicts=["查订单", "问政策"],
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert "冲突" in " ".join(result.issues)

    def test_direction_invalid_bare(self, validator):
        """Even without missing_info or conflicts, invalid is invalid."""
        plan = make_plan(
            direction="invalid",
            reason="无法理解",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False


# ===================================================================
# Task-specific validation failures
# ===================================================================


class TestTaskValidationFailures:
    def test_task_missing_flow_id(self, validator):
        plan = make_plan(direction="task", flow_id=None, action="start")
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("未指定 flow_id" in i for i in result.issues)

    def test_task_unknown_flow_id(self, validator):
        plan = make_plan(direction="task", flow_id="nonexistent_flow", action="start")
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("不在已知流程中" in i for i in result.issues)

    def test_task_resume_not_in_paused(self, validator):
        """Resume action but the flow_id is not in paused_tasks."""
        plan = make_plan(direction="task", flow_id="refund_request", action="resume")
        result = validator.validate(plan, make_state())  # no paused_tasks
        assert result.is_valid is False
        assert any("不在暂停任务列表中" in i for i in result.issues)

    def test_task_cancel_no_active(self, validator):
        """Cancel action but no active_task."""
        plan = make_plan(
            direction="task", flow_id="order_status_query", action="cancel",
        )
        result = validator.validate(plan, make_state())  # no active_task
        assert result.is_valid is False
        assert any("没有 active_task" in i for i in result.issues)

    def test_task_cancel_wrong_flow(self, validator):
        """Cancel action on a different flow than the active one."""
        state = make_state(
            active_task=TaskContext(flow_id="logistics_tracking", step_id="lookup"),
        )
        plan = make_plan(direction="task", flow_id="refund_request", action="cancel")
        result = validator.validate(plan, state)
        assert result.is_valid is False
        assert any("当前 active_task" in i for i in result.issues)


# ===================================================================
# Knowledge-specific validation failures
# ===================================================================


class TestKnowledgeValidationFailures:
    def test_knowledge_missing_intent(self, validator):
        plan = make_plan(
            direction="knowledge",
            knowledge_intent=None,
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("未指定 knowledge_intent" in i for i in result.issues)

    def test_knowledge_unknown_intent(self, validator):
        plan = make_plan(
            direction="knowledge",
            knowledge_intent="天文知识",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("不在已知知识意图中" in i for i in result.issues)


# ===================================================================
# Object reference checks
# ===================================================================


class TestObjectReference:
    def test_pronoun_without_focused_object(self, validator):
        """Plan reason contains '它' but state has no focused_object."""
        plan = make_plan(
            direction="task",
            flow_id="logistics_tracking",
            action="start",
            reason="用户想查它的物流",
        )
        result = validator.validate(plan, make_state())  # no focused_object
        assert result.is_valid is False
        assert any("没有 focused_object" in i for i in result.issues)

    def test_pronoun_in_user_text_without_focused_object(self, validator):
        """User text contains '这个' but state has no focused_object."""
        plan = make_plan(
            direction="task",
            flow_id="order_status_query",
            action="start",
            reason="用户想查订单",
        )
        result = validator.validate(
            plan, make_state(), user_text="这个订单什么时候到？",
        )
        assert result.is_valid is False
        assert any("没有 focused_object" in i for i in result.issues)

    def test_pronoun_with_focused_object_is_ok(self, validator):
        """When focused_object exists, pronouns are fine."""
        state = make_state(
            focused_object=FocusedObject(
                type="order", id="ORD_1", title="订单 ORD_1",
            ),
        )
        plan = make_plan(
            direction="task",
            flow_id="logistics_tracking",
            action="start",
            reason="用户想查它的物流",
        )
        result = validator.validate(plan, state, user_text="帮我查它的物流")
        assert result.is_valid is True


# ===================================================================
# Internal consistency checks
# ===================================================================


class TestInternalConsistency:
    def test_empty_reason(self, validator):
        plan = make_plan(
            direction="chitchat",
            reason="",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("缺少 reason" in i for i in result.issues)

    def test_task_with_conflicts(self, validator):
        """A task plan that also lists conflicts is suspicious."""
        plan = make_plan(
            direction="task",
            flow_id="logistics_tracking",
            action="start",
            conflicts=["可能也是退款"],
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("同时报告了冲突" in i for i in result.issues)

    def test_chitchat_with_task_details(self, validator):
        """Chitchat shouldn't carry task parameters."""
        plan = make_plan(
            direction="chitchat",
            flow_id="logistics_tracking",
            knowledge_intent=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("携带了 flow_id" in i for i in result.issues)

    def test_chitchat_with_knowledge_details(self, validator):
        plan = make_plan(
            direction="chitchat",
            knowledge_intent="商品信息",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("携带了 flow_id 或 knowledge_intent" in i for i in result.issues)

    def test_reason_whitespace_only(self, validator):
        plan = make_plan(
            direction="chitchat",
            reason="   ",
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is False
        assert any("缺少 reason" in i for i in result.issues)


# ===================================================================
# Validator with None user_text
# ===================================================================


class TestDegradedWithoutUserText:
    def test_validator_works_without_user_text(self, validator):
        """Validator degrades gracefully when user_text is not provided."""
        plan = make_plan(direction="chitchat", reason="打招呼", flow_id=None, action=None)
        result = validator.validate(plan, make_state(), user_text=None)
        assert result.is_valid is True

    def test_object_reference_skipped_without_user_text(self, validator):
        """Without user_text, can't check for pronouns there — but plan reason
        is still checked."""
        plan = make_plan(
            direction="task",
            flow_id="logistics_tracking",
            action="start",
            reason="用户想查物流",  # no pronoun in reason
        )
        result = validator.validate(
            plan, make_state(), user_text=None,
        )
        # Should pass because reason doesn't contain pronoun markers, and
        # user_text=None means we skip user_text pronoun check.
        assert result.is_valid is True


# ===================================================================
# All known flows pass basic validation
# ===================================================================


class TestAllKnownFlows:
    @pytest.mark.parametrize("flow_id", [
        "onboarding",
        "order_status_query",
        "logistics_tracking",
        "refund_request",
        "similar_product_recommendation",
        "human_handoff",
    ])
    def test_every_known_flow_is_valid(self, validator, flow_id):
        plan = make_plan(direction="task", flow_id=flow_id, action="start")
        result = validator.validate(plan, make_state())
        assert result.is_valid is True, f"{flow_id} should be valid"

    @pytest.mark.parametrize("intent", ["商品信息", "退换货政策", "常见问题"])
    def test_every_known_intent_is_valid(self, validator, intent):
        plan = make_plan(
            direction="knowledge",
            knowledge_intent=intent,
            flow_id=None,
            action=None,
        )
        result = validator.validate(plan, make_state())
        assert result.is_valid is True, f"{intent} should be valid"


# ===================================================================
# ValidationResult schema
# ===================================================================


class TestValidationResultSchema:
    def test_valid_result_has_direction(self):
        result = ValidationResult(is_valid=True, direction="task")
        assert result.is_valid is True
        assert result.direction == "task"
        assert result.issues == []

    def test_invalid_result_has_no_direction(self):
        result = ValidationResult(is_valid=False, issues=["意图不明确"])
        assert result.is_valid is False
        assert result.direction is None

    def test_invalid_result_accumulates_issues(self):
        result = ValidationResult(
            is_valid=False,
            issues=["意图不明确", "缺少对象", "多个方向冲突"],
        )
        assert len(result.issues) == 3


# ===================================================================
# TurnPlan schema
# ===================================================================


class TestTurnPlanSchema:
    def test_minimal_task_plan(self):
        plan = TurnPlan(direction="task", flow_id="order_status_query", action="start")
        assert plan.direction == "task"
        assert plan.reason == ""
        assert plan.knowledge_intent is None
        assert plan.missing_info is None
        assert plan.conflicts == []

    def test_invalid_plan_defaults(self):
        plan = TurnPlan(direction="invalid")
        assert plan.direction == "invalid"
        assert plan.flow_id is None
        assert plan.action is None
        assert plan.conflicts == []

    def test_invalid_plan_with_details(self):
        plan = TurnPlan(
            direction="invalid",
            reason="用户意图不明确",
            missing_info="用户说'这个怎么办'但缺少具体对象",
            conflicts=["可能查订单", "可能退款"],
        )
        assert plan.missing_info is not None
        assert len(plan.conflicts) == 2