"""TurnPlanValidator — checks TurnPlanner output before execution.

Step 6 of the DialogueEngine design (§2.6): the planner's raw result must
be validated before it is acted on.  When validation fails the engine routes
to ClarifyResponder instead of executing a handler.
"""

from __future__ import annotations

from typing import Optional

from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    TurnPlan,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Known capabilities — mirrors flow_config/ and the system prompt in engine.py.
# When flows become runtime-loaded these constants will be replaced by a
# FlowRegistry lookup.
# ---------------------------------------------------------------------------

KNOWN_FLOWS: set[str] = {
    "onboarding",
    "order_status_query",
    "logistics_tracking",
    "refund_request",
    "similar_product_recommendation",
    "human_handoff",
}

KNOWN_KNOWLEDGE_INTENTS: set[str] = {
    "商品信息",
    "退换货政策",
    "常见问题",
}


class TurnPlanValidator:
    """Validate a TurnPlan before the engine commits to a direction.

    Checks are intentionally strict — a plan that passes validation should
    be safe to route directly to a handler.  Plans that fail validation
    produce a ``ValidationResult`` whose ``issues`` list explains every
    reason, so ClarifyResponder can generate a targeted follow-up.
    """

    # -- public API ------------------------------------------------------------

    def validate(
        self,
        plan: TurnPlan,
        state: DialogueState,
        user_text: Optional[str] = None,
    ) -> ValidationResult:
        """Check *plan* against *state* and return a ``ValidationResult``.

        Parameters
        ----------
        plan:
            The structured planning result from TurnPlanner.
        state:
            The full dialogue state for context-aware checks (active task,
            paused tasks, focused object).
        user_text:
            The raw user message text.  Used for heuristic checks (e.g.
            pronoun-without-referent).  Optional — the validator degrades
            gracefully when it is not available.
        """
        issues: list[str] = []

        # 1. Planner self-reported inability to understand
        if plan.direction == "invalid":
            issues.append("TurnPlanner 无法确定用户意图")
            if plan.missing_info:
                issues.append(f"缺少信息：{plan.missing_info}")
            if plan.conflicts:
                for c in plan.conflicts:
                    issues.append(f"冲突：{c}")
            return ValidationResult(is_valid=False, direction=None, issues=issues)

        # 2. Direction-specific structural checks
        if plan.direction == "task":
            _check_task_plan(plan, state, issues)
        elif plan.direction == "knowledge":
            _check_knowledge_plan(plan, issues)
        elif plan.direction == "chitchat":
            # Chitchat is the lowest-risk direction — always structurally valid.
            pass

        # 3. Cross-cutting heuristic checks
        _check_object_reference(plan, state, user_text, issues)
        _check_internal_consistency(plan, issues)

        if issues:
            return ValidationResult(is_valid=False, direction=None, issues=issues)

        return ValidationResult(
            is_valid=True,
            direction=plan.direction,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Direction-specific checks
# ---------------------------------------------------------------------------


def _check_task_plan(plan: TurnPlan, state: DialogueState, issues: list[str]) -> None:
    """Validate a task-direction plan against available flows and state."""
    if not plan.flow_id:
        issues.append("计划方向为 task 但未指定 flow_id")
        return

    if plan.flow_id not in KNOWN_FLOWS:
        issues.append(
            f"指定的 flow_id \"{plan.flow_id}\" 不在已知流程中。"
            f"已知流程：{', '.join(sorted(KNOWN_FLOWS))}"
        )

    # Action-specific state checks
    if plan.action == "resume":
        paused_ids = {t.flow_id for t in state.paused_tasks}
        if plan.flow_id not in paused_ids:
            issues.append(
                f"计划操作为 resume，但 \"{plan.flow_id}\" 不在暂停任务列表中"
            )
    elif plan.action == "cancel":
        active = state.active_task
        if active is None:
            issues.append("计划操作为 cancel，但当前没有 active_task")
        elif active.flow_id != plan.flow_id:
            issues.append(
                f"计划取消 \"{plan.flow_id}\"，但当前 active_task 是 "
                f"\"{active.flow_id}\""
            )
    elif plan.action == "continue":
        active = state.active_task
        if active is None:
            issues.append("计划操作为 continue，但当前没有 active_task")


def _check_knowledge_plan(plan: TurnPlan, issues: list[str]) -> None:
    """Validate a knowledge-direction plan."""
    if not plan.knowledge_intent:
        issues.append("计划方向为 knowledge 但未指定 knowledge_intent")
        return

    if plan.knowledge_intent not in KNOWN_KNOWLEDGE_INTENTS:
        issues.append(
            f"knowledge_intent \"{plan.knowledge_intent}\" 不在已知知识意图中。"
            f"已知意图：{', '.join(sorted(KNOWN_KNOWLEDGE_INTENTS))}"
        )


# ---------------------------------------------------------------------------
# Cross-cutting heuristic checks
# ---------------------------------------------------------------------------


def _check_object_reference(
    plan: TurnPlan,
    state: DialogueState,
    user_text: Optional[str],
    issues: list[str],
) -> None:
    """Flag plans that appear to reference an object the state doesn't have.

    This is a heuristic — it won't catch every case, but it prevents the most
    common failure mode: the user says "它"/"这个" without any focused_object.
    """
    # If a focused object is already present, no problem.
    if state.focused_object is not None:
        return

    # Check the plan reason for object-deictic language.
    pronoun_markers = ("它", "这个", "该商品", "该订单", "这个商品", "这个订单")
    reason_has_pronoun = any(m in plan.reason for m in pronoun_markers)

    # Also check the raw user text if available.
    text_has_pronoun = False
    if user_text:
        text_has_pronoun = any(m in user_text for m in pronoun_markers)

    if reason_has_pronoun or text_has_pronoun:
        issues.append(
            '用户消息引用了对象（如“它”、“这个”），但当前没有 focused_object。'
            '无法确定用户具体指的是哪个订单或商品。'
        )


def _check_internal_consistency(plan: TurnPlan, issues: list[str]) -> None:
    """Check for contradictions within the plan itself."""
    if not plan.reason.strip():
        issues.append("计划缺少 reason，无法验证其推理依据")

    # A task plan that claims to resume but also lists conflicts is suspicious.
    if plan.direction == "task" and plan.conflicts:
        issues.append(
            f"计划方向是 task 但同时报告了冲突：{'；'.join(plan.conflicts)}"
        )

    # A chitchat plan that carries task details is suspicious.
    if plan.direction == "chitchat" and (plan.flow_id or plan.knowledge_intent):
        issues.append("计划方向是 chitchat 但携带了 flow_id 或 knowledge_intent")