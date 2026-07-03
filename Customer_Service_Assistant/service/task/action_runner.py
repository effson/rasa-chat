"""ActionRunner — execute actions declared in flow steps.

This is step 4 of the TaskHandler pipeline.  ActionRunner receives an
:class:`ActionFlowStep`, dispatches to the appropriate handler (built-in
or custom), and returns an :class:`ActionResult`.

Built-in actions
----------------
``action_response``
    Reply to the user.  Supports three modes:

    * ``static`` — render *text* template directly.
    * ``rephrase`` — render *text* as a suggestion, rewrite via LLM.
    * ``generate`` — generate from scratch via LLM using *prompt*.

``action_listen``
    Signal that the flow should pause and wait for user input.

Custom actions
--------------
Custom actions (e.g. ``action_lookup_order_status``) are registered via
:class:`ActionRegistry` and receive the step args plus the full dialogue
state so they can read slots and update them in return.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from Customer_Service_Assistant.infrastructure.llm import llm as _default_llm
from Customer_Service_Assistant.service.schemas import DialogueState, Message, SystemContext
from Customer_Service_Assistant.service.task.steps import ActionFlowStep


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActionResult:
    """The result of executing a single action.

    Attributes:
        messages: Replies to append to the current turn.
        slot_updates: Key-value pairs to write into ``active_task.slots``.
        should_listen: When ``True``, the flow executor should stop and
            wait for the next user input (i.e. ``action_listen`` was hit).
    """

    messages: List[Message] = field(default_factory=list)
    slot_updates: Dict[str, Any] = field(default_factory=dict)
    should_listen: bool = False

    @classmethod
    def listen(cls) -> "ActionResult":
        """Factory for an ``action_listen`` result."""
        return cls(should_listen=True)

    @classmethod
    def reply(cls, text: str) -> "ActionResult":
        """Factory for a single bot-message result."""
        return cls(messages=[Message(role="bot", text=text)])


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

# Matches ``{{ slots.xxx }}``, ``{{ context.xxx }}`` and ``{{ history }}``
_TEMPLATE_RE = re.compile(r"\{\{\s*(\w+)(?:\.(\w+))?\s*\}\}")


def render_template(
    text: str,
    *,
    slots: Dict[str, Any] | None = None,
    context: SystemContext | None = None,
    extra: Dict[str, Any] | None = None,
) -> str:
    """Replace ``{{ slots.key }}``, ``{{ context.key }}`` and ``{{ bare_key }}``
    placeholders in *text* with their values.

    * ``{{ slots.xxx }}`` → ``slots["xxx"]`` (task slots)
    * ``{{ context.xxx }}`` → ``getattr(system_context, "xxx")``
    * ``{{ bare_key }}`` → ``extra["bare_key"]``  (history, user_message, ...)
    """
    slots = slots or {}
    extra = extra or {}

    def _replace(match: re.Match) -> str:
        namespace = match.group(1)
        key = match.group(2)

        # Two-segment: {{ slots.xxx }} or {{ context.xxx }}
        if key is not None:
            if namespace == "slots":
                val = slots.get(key)
                if val is not None:
                    return str(val)
                return match.group(0)

            if namespace == "context":
                if context is not None and hasattr(context, key):
                    val = getattr(context, key, None)
                    if val is not None:
                        return str(val)
                return match.group(0)

            return match.group(0)

        # Single-segment: {{ history }}, {{ user_message }}, etc.
        if namespace in extra:
            return str(extra[namespace])
        return match.group(0)

    return re.sub(_TEMPLATE_RE, _replace, text)


# ---------------------------------------------------------------------------
# ActionRunner
# ---------------------------------------------------------------------------


class ActionRunner:
    """Execute an :class:`ActionFlowStep` and return an :class:`ActionResult`.

    Parameters
    ----------
    llm:
        The LangChain chat model used for ``rephrase`` / ``generate`` modes.
        Defaults to the project-wide singleton.
    registry:
        An :class:`ActionRegistry` mapping custom action names to handlers.
        When ``None``, no custom actions are available.
    """

    def __init__(
        self,
        llm=None,
        registry: ActionRegistry | None = None,
    ) -> None:
        self._llm = llm or _default_llm
        self._registry = registry if registry is not None else create_default_registry()

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        step: ActionFlowStep,
        state: DialogueState,
    ) -> ActionResult:
        """Execute *step* in the context of *state*."""
        action = step.action

        if action == "action_response":
            return await self._handle_response(step, state)

        if action == "action_listen":
            return ActionResult.listen()

        # Try custom action
        handler = self._registry.get(action)
        if handler is not None:
            return await handler(step.args, state)

        # Unknown action — return empty so executor can skip gracefully
        return ActionResult()

    # -- action_response -----------------------------------------------------

    async def _handle_response(
        self,
        step: ActionFlowStep,
        state: DialogueState,
    ) -> ActionResult:
        # Resolve args — may be a string reference like "context.response"
        args = self._resolve_args(step.args, state)

        mode = args.get("mode", "static")
        text = args.get("text")
        prompt = args.get("prompt")

        # Build the template variable environment
        slots = state.active_task.slots if state.active_task else {}
        system_ctx = state.active_system_flow

        if mode == "static":
            rendered = render_template(text or "", slots=slots, context=system_ctx)
            return ActionResult.reply(rendered)

        if mode == "rephrase":
            current_response = render_template(
                text or "", slots=slots, context=system_ctx
            )
            llm_text = await self._llm_rephrase(
                prompt=prompt or "",
                current_response=current_response,
                state=state,
                slots=slots,
                context=system_ctx,
            )
            return ActionResult.reply(llm_text)

        if mode == "generate":
            llm_text = await self._llm_generate(
                prompt=prompt or "",
                state=state,
                slots=slots,
                context=system_ctx,
            )
            return ActionResult.reply(llm_text)

        # Unknown mode — fall back to static rendering
        rendered = render_template(text or "", slots=slots, context=system_ctx)
        return ActionResult.reply(rendered)

    # -- LLM helpers ---------------------------------------------------------

    async def _llm_rephrase(
        self,
        prompt: str,
        current_response: str,
        state: DialogueState,
        slots: Dict[str, Any],
        context: SystemContext | None,
    ) -> str:
        extra = {
            "current_response": current_response,
            "user_message": self._get_user_text(state),
            "history": self._get_history(state),
        }
        rendered_prompt = render_template(
            prompt, slots=slots, context=context, extra=extra
        )
        response = await self._llm.ainvoke([{"role": "user", "content": rendered_prompt}])
        return (response.content or "").strip() if response.content else current_response

    async def _llm_generate(
        self,
        prompt: str,
        state: DialogueState,
        slots: Dict[str, Any],
        context: SystemContext | None,
    ) -> str:
        extra = {
            "user_message": self._get_user_text(state),
            "history": self._get_history(state),
        }
        rendered_prompt = render_template(
            prompt, slots=slots, context=context, extra=extra
        )
        response = await self._llm.ainvoke([{"role": "user", "content": rendered_prompt}])
        return (response.content or "").strip() if response.content else ""

    # -- helpers -------------------------------------------------------------

    def _resolve_args(
        self,
        args: Dict[str, Any] | str,
        state: DialogueState,
    ) -> Dict[str, Any]:
        """Resolve *args* to a plain dict.

        When *args* is a string like ``"context.response"`` it refers to
        ``state.active_system_flow.response`` (a dict).  This is used by
        ``system_collect_information`` to forward the collect step's
        ``response`` block.
        """
        if isinstance(args, dict):
            return args

        # String reference — e.g. "context.response"
        if isinstance(args, str):
            parts = args.split(".", 1)
            if parts[0] == "context" and len(parts) == 2:
                ctx = state.active_system_flow
                if ctx is not None and hasattr(ctx, parts[1]):
                    return getattr(ctx, parts[1], {}) or {}
            return {}

        return {}

    @staticmethod
    def _get_user_text(state: DialogueState) -> str:
        if state.pending_turn and state.pending_turn.input_message.text:
            return state.pending_turn.input_message.text
        return ""

    @staticmethod
    def _get_history(state: DialogueState) -> str:
        """Format recent conversation as a readable history string."""
        lines: List[str] = []
        for msg in state.current_messages:
            role = "用户" if msg.role == "user" else "客服"
            text = msg.text or ""
            lines.append(f"{role}：{text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Custom action protocol
# ---------------------------------------------------------------------------


class CustomAction(Protocol):
    """Protocol for a custom action handler.

    A custom action receives the step's *args* dict and the full dialogue
    state, and returns an :class:`ActionResult`.
    """

    async def __call__(
        self,
        args: Dict[str, Any],
        state: DialogueState,
    ) -> ActionResult: ...


class ActionRegistry:
    """Registry of custom action handlers keyed by action name.

    Usage::

        registry = ActionRegistry()
        registry.register("action_lookup_order_status", lookup_handler)

        handler = registry.get("action_lookup_order_status")
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, CustomAction] = {}

    def register(self, name: str, handler: CustomAction) -> None:
        """Register a custom action *handler* under *name*."""
        self._handlers[name] = handler

    def get(self, name: str) -> CustomAction | None:
        """Return the handler for *name*, or ``None``."""
        return self._handlers.get(name)


# ---------------------------------------------------------------------------
# Default custom action stubs
# ---------------------------------------------------------------------------


async def _lookup_order_status(
    args: Dict[str, Any],
    state: DialogueState,
) -> ActionResult:
    """Stub for ``action_lookup_order_status``.

    Reads ``order_number`` from active task slots and returns placeholder
    status data.  Will be replaced by a real commerce-API call later.
    """
    order_number = ""
    if state.active_task:
        order_number = str(state.active_task.slots.get("order_number", ""))
    return ActionResult(
        slot_updates={
            "order_status": "已发货",
            "order_summary": f"订单{order_number}正在配送中，预计明天送达",
        }
    )


async def _lookup_logistics(
    args: Dict[str, Any],
    state: DialogueState,
) -> ActionResult:
    """Stub for ``action_lookup_logistics``.

    Reads ``order_number`` from active task slots and returns placeholder
    logistics data.  Will be replaced by a real commerce-API call later.
    """
    order_number = ""
    if state.active_task:
        order_number = str(state.active_task.slots.get("order_number", ""))
    return ActionResult(
        slot_updates={
            "logistics_company": "顺丰速运",
            "tracking_number": f"SF{order_number}",
            "logistics_status": "运输中，预计明天送达",
        }
    )


async def _recommend_similar_products(
    args: Dict[str, Any],
    state: DialogueState,
) -> ActionResult:
    """Stub for ``action_recommend_similar_products``.

    Reads ``product_id`` from active task slots and returns a placeholder
    recommendation message.  Will be replaced by a real commerce-API call
    later.
    """
    product_id = ""
    if state.active_task:
        product_id = str(state.active_task.slots.get("product_id", ""))
    return ActionResult.reply(
        f"根据商品{product_id}，为您推荐以下相似商品：..."
    )


def create_default_registry() -> ActionRegistry:
    """Return an ``ActionRegistry`` pre-populated with the three project
    custom actions (stub implementations)."""
    registry = ActionRegistry()
    registry.register("action_lookup_order_status", _lookup_order_status)
    registry.register("action_lookup_logistics", _lookup_logistics)
    registry.register("action_recommend_similar_products", _recommend_similar_products)
    return registry