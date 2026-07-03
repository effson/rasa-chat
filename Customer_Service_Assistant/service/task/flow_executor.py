"""FlowExecutor — advance through flows and execute actions.

This is step 5 of the TaskHandler pipeline (§4.3 of the design doc).
FlowExecutor has two layers:

* **Outer loop** (``run``) — repeatedly gets the next action from the
  inner loop, executes it via :class:`ActionRunner`, applies slot
  updates, and collects reply messages.  Stops when ``action_listen``
  is hit or there is no more work to do.

* **Inner loop** (``_advance_until_action``) — walks the current flow
  (system or business) step by step, handling ``start`` / ``collect`` /
  ``end`` transparently, and only returns when it hits an ``action``
  step (or runs out of steps).

Step handling summary
---------------------
+-----------+-----------------------------------------------+
| Step type | Behaviour                                     |
+-----------+-----------------------------------------------+
| start     | Move to next step immediately.                |
+-----------+-----------------------------------------------+
| collect   | Auto-fill from focused_object; if the slot    |
|           | has a value → validate → advance; otherwise   |
|           | activate ``system_collect_information``.      |
+-----------+-----------------------------------------------+
| action    | Update step_id to next, return action to      |
|           | outer loop.                                   |
+-----------+-----------------------------------------------+
| end       | End current flow (system → clear; business    |
|           | → activate ``system_completed``).             |
+-----------+-----------------------------------------------+
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from Customer_Service_Assistant.service.schemas import (
    CollectSystemContext,
    CompletedSystemContext,
    DialogueState,
    Message,
)
from Customer_Service_Assistant.service.task.action_runner import ActionRunner
from Customer_Service_Assistant.service.task.links import (
    ConditionalLink,
    FallbackLink,
    FlowStepLink,
    StaticLink,
)
from Customer_Service_Assistant.service.task.models import Flow, FlowsList
from Customer_Service_Assistant.service.task.steps import (
    ActionFlowStep,
    CollectSlotStep,
    EndFlowStep,
    FlowStep,
    FlowStepType,
    ResponseDefinition,
    StartFlowStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response_to_dict(resp: ResponseDefinition) -> Dict[str, object]:
    """Convert a ``ResponseDefinition`` to a plain dict for
    ``CollectSystemContext.response``."""
    d: Dict[str, object] = {"mode": resp.mode}
    if resp.text is not None:
        d["text"] = resp.text
    if resp.prompt is not None:
        d["prompt"] = resp.prompt
    return d


def _context_to_dict(ctx: object | None) -> Dict[str, object]:
    """Convert a system context (Pydantic model or None) to a plain dict
    for use in condition evaluation."""
    if ctx is None:
        return {}
    if hasattr(ctx, "model_dump"):
        return ctx.model_dump()  # type: ignore[union-attr]
    return {}


# ---------------------------------------------------------------------------
# FlowExecutor
# ---------------------------------------------------------------------------


class FlowExecutor:
    """Advance flows and execute actions until ``action_listen``.

    Parameters
    ----------
    action_runner:
        The :class:`ActionRunner` used to execute action steps.
    """

    def __init__(self, action_runner: ActionRunner) -> None:
        self._action_runner = action_runner

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        state: DialogueState,
        flows: FlowsList,
    ) -> List[Message]:
        """Execute flows until the next ``action_listen`` (or no work remains).

        Returns the list of bot reply messages produced this round.
        """
        messages: List[Message] = []

        while True:
            action_step = self._advance_until_action(state, flows)
            if action_step is None:
                break  # action_listen or nothing to do

            result = await self._action_runner.run(action_step, state)

            # Write slot updates back to state
            for name, value in result.slot_updates.items():
                state.set_slot(name, value)

            messages.extend(result.messages)

            if result.should_listen:
                break

        return messages

    # -- inner loop ----------------------------------------------------------

    def _advance_until_action(
        self,
        state: DialogueState,
        flows: FlowsList,
    ) -> ActionFlowStep | None:
        """Walk the current flow until an action step is reached.

        Returns ``None`` to signal ``action_listen`` (no more work).
        """
        while True:
            # 1. Determine which flow/step to process
            flow, step_id = self._current_context(state, flows)
            if flow is None or step_id is None:
                return None

            step = flow.get_step(step_id)
            if step is None:
                return None

            # 2. Dispatch by step type
            if isinstance(step, StartFlowStep):
                next_id = self._resolve_next(step, state)
                self._set_step_id(state, next_id)
                continue

            elif isinstance(step, CollectSlotStep):
                action = self._handle_collect(step, state, flows)
                if action is not None:
                    return action
                continue

            elif isinstance(step, ActionFlowStep):
                next_id = self._resolve_next(step, state)
                self._set_step_id(state, next_id)
                return step

            elif isinstance(step, EndFlowStep):
                self._handle_end(state, flows)
                continue

            else:
                # Unknown step type — advance if possible
                next_id = self._resolve_next(step, state)
                self._set_step_id(state, next_id)
                continue

    # -- current context -----------------------------------------------------

    def _current_context(
        self,
        state: DialogueState,
        flows: FlowsList,
    ) -> tuple[Flow | None, str | None]:
        """Return ``(flow, step_id)`` for the currently active context.

        Priority: system flow > business task > nothing.
        """
        ctx = state.active_system_flow
        if ctx is not None and ctx.step_id is not None:
            flow = flows.get_flow(ctx.flow_id)
            return flow, ctx.step_id

        task = state.active_task
        if task is not None and task.step_id is not None:
            flow = flows.get_flow(task.flow_id)
            return flow, task.step_id

        return None, None

    def _set_step_id(self, state: DialogueState, step_id: str | None) -> None:
        """Write *step_id* into whichever context (system or task) is active."""
        if step_id is None:
            return
        if state.active_system_flow is not None:
            state.active_system_flow.step_id = step_id
        elif state.active_task is not None:
            state.active_task.step_id = step_id

    # -- next-link resolution ------------------------------------------------

    def _resolve_next(self, step: FlowStep, state: DialogueState) -> str | None:
        """Evaluate *step.next* links in order and return the first match."""
        if not step.next:
            return None

        slots = state.active_task.slots if state.active_task else {}
        ctx_dict = _context_to_dict(state.active_system_flow)

        for link in step.next:
            if isinstance(link, StaticLink):
                return link.target
            elif isinstance(link, ConditionalLink):
                if self._evaluate_condition(link.condition, slots, ctx_dict):
                    return link.target
            elif isinstance(link, FallbackLink):
                return link.target

        return None

    def _evaluate_condition(
        self,
        condition: str,
        slots: Dict[str, object],
        context: Dict[str, object],
    ) -> bool:
        """Safely evaluate a Python condition expression.

        Only *slots* and *context* are in scope; all builtins are disabled.
        """
        try:
            result = eval(
                condition,
                {"__builtins__": {}},
                {"slots": slots, "context": context},
            )
            return bool(result)
        except Exception:
            return False

    # -- collect step --------------------------------------------------------

    def _handle_collect(
        self,
        step: CollectSlotStep,
        state: DialogueState,
        flows: FlowsList,
    ) -> ActionFlowStep | None:
        slot_name = step.slot_name

        # 1. Auto-fill from focused_object
        self._auto_fill_slot(step, state)

        # 2. Check if slot has a value
        active = state.active_task
        slot_value = active.slots.get(slot_name) if active else None

        if slot_value is not None and slot_value != "":
            # 3. Validate
            if step.validation and step.validation.condition:
                ctx_dict = _context_to_dict(state.active_system_flow)
                if not self._evaluate_condition(
                    step.validation.condition,
                    active.slots if active else {},
                    ctx_dict,
                ):
                    # Validation failed — clear slot
                    if active:
                        active.slots.pop(slot_name, None)
                    # If there's a failure response, return it as an action
                    if step.validation.failure_response is not None:
                        return self._failure_response_action(
                            step, step.validation.failure_response
                        )
                    # No failure response — loop back (re-collect)
                    return None

            # Slot OK — advance
            next_id = self._resolve_next(step, state)
            self._set_step_id(state, next_id)
            return None

        # 4. Slot missing — activate system_collect_information
        system_flow = flows.get_flow("system_collect_information")
        if system_flow is not None:
            start = system_flow.get_start_step()
            state.activate_system_flow(
                CollectSystemContext(
                    slot_name=slot_name,
                    response=_response_to_dict(step.response),
                    step_id=start.id if start else None,
                )
            )
            return None

        # system_collect_information not in FlowsList — advance anyway
        # to avoid an infinite loop.
        next_id = self._resolve_next(step, state)
        self._set_step_id(state, next_id)
        return None

    def _auto_fill_slot(
        self,
        step: CollectSlotStep,
        state: DialogueState,
    ) -> None:
        """Try to auto-fill *step.slot_name* from ``focused_object``."""
        if state.active_task is None:
            return

        slot_name = step.slot_name

        # Already filled
        if slot_name in state.active_task.slots and state.active_task.slots[slot_name]:
            return

        focus = state.focused_object
        if focus is None:
            return

        # Map object type → slot name
        type_to_slot: Dict[str, str] = {
            "order": "order_number",
            "product": "product_id",
        }
        for obj_type, mapped_slot in type_to_slot.items():
            if focus.type == obj_type and slot_name == mapped_slot:
                state.set_slot(slot_name, focus.id)
                return

    def _failure_response_action(
        self,
        step: CollectSlotStep,
        failure: ResponseDefinition,
    ) -> ActionFlowStep:
        """Build a synthetic ``action_response`` step for a validation failure."""
        return ActionFlowStep(
            id=f"{step.id}_validation_failed",
            type=FlowStepType.ACTION,
            action="action_response",
            args=_response_to_dict(failure),
        )

    # -- end step -------------------------------------------------------------

    def _handle_end(
        self,
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        """End the current flow (system or business)."""
        if state.active_system_flow is not None:
            state.active_system_flow = None
            return

        if state.active_task is not None:
            state.active_task = None
            # Activate system_completed
            completed_flow = flows.get_flow("system_completed")
            if completed_flow is not None:
                start = completed_flow.get_start_step()
                state.activate_system_flow(
                    CompletedSystemContext(
                        step_id=start.id if start else None,
                    )
                )