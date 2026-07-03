"""CommandProcessor — apply :class:`Command` objects to :class:`DialogueState`.

This is step 3 of the TaskHandler pipeline (§4 of the design doc).
It receives parsed commands and updates the dialogue state's task context
before :class:`FlowExecutor` resumes flow execution.

Entry point::

    processor = CommandProcessor()
    processor.run(commands, state, flows)
"""

from __future__ import annotations

from typing import List

from Customer_Service_Assistant.service.schemas import (
    CanceledSystemContext,
    DialogueState,
    InterruptedSystemContext,
    ResumedSystemContext,
    StartedSystemContext,
)
from Customer_Service_Assistant.service.task.commands import (
    CancelFlowCommand,
    Command,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)
from Customer_Service_Assistant.service.task.models import Flow, FlowsList


class CommandProcessorError(Exception):
    """Raised when a command cannot be applied (e.g. unknown flow_id)."""

    pass


class CommandProcessor:
    """Apply a list of commands to a dialogue state, mutating task context."""

    # -- public API ----------------------------------------------------------

    def run(
        self,
        commands: List[Command],
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        """Apply each command in *commands* to *state*.

        Commands are applied in order.  Early commands (e.g. ``cancel_flow``)
        change the state that later commands (e.g. ``start_flow``) see.
        """
        for command in commands:
            self._apply(command, state, flows)

    # -- dispatch ------------------------------------------------------------

    def _apply(
        self,
        command: Command,
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        if isinstance(command, StartFlowCommand):
            self._handle_start_flow(command, state, flows)
        elif isinstance(command, SetSlotsCommand):
            self._handle_set_slots(command, state)
        elif isinstance(command, CancelFlowCommand):
            self._handle_cancel_flow(state, flows)
        elif isinstance(command, ResumeFlowCommand):
            self._handle_resume_flow(command, state, flows)

    # -- start_flow ----------------------------------------------------------

    def _handle_start_flow(
        self,
        command: StartFlowCommand,
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        target_flow_id = command.flow
        target_flow = self._get_flow_or_raise(target_flow_id, flows)
        start_step = target_flow.get_start_step()
        if start_step is None:
            raise CommandProcessorError(
                f"Flow '{target_flow_id}' 缺少 start step"
            )

        # Always clear the system flow before starting a new task.
        state.active_system_flow = None

        active = state.active_task

        # Case 1 — no active task: start fresh.
        if active is None:
            state.start_new_task(target_flow_id, start_step.id)
            self._activate_system(
                state,
                flows,
                "system_task_started",
                StartedSystemContext(
                    started_flow_id=target_flow_id,
                    started_flow_name=target_flow.name or target_flow_id,
                ),
            )
            return

        # Case 2 — already on the target flow: no-op.
        if active.flow_id == target_flow_id:
            return

        # Case 3 — different task is active: pause it, start new one.
        old_flow = flows.get_flow(active.flow_id)
        old_name = old_flow.name if old_flow else active.flow_id
        state.pause_active_task()
        state.start_new_task(target_flow_id, start_step.id)

        self._activate_system(
            state,
            flows,
            "system_task_interrupted",
            InterruptedSystemContext(
                interrupted_flow_id=active.flow_id,
                interrupted_flow_name=old_name,
                started_flow_id=target_flow_id,
                started_flow_name=target_flow.name or target_flow_id,
            ),
        )

    # -- set_slots -----------------------------------------------------------

    def _handle_set_slots(
        self,
        command: SetSlotsCommand,
        state: DialogueState,
    ) -> None:
        if state.active_task is None:
            return
        for name, value in command.slots.items():
            state.set_slot(name, value)

    # -- cancel_flow ---------------------------------------------------------

    def _handle_cancel_flow(
        self,
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        canceled = state.cancel_active_task()
        if canceled is None:
            return

        old_flow = flows.get_flow(canceled.flow_id)
        canceled_name = old_flow.name if old_flow else canceled.flow_id

        self._activate_system(
            state,
            flows,
            "system_task_canceled",
            CanceledSystemContext(
                canceled_flow_id=canceled.flow_id,
                canceled_flow_name=canceled_name,
            ),
        )

    # -- resume_flow ---------------------------------------------------------

    def _handle_resume_flow(
        self,
        command: ResumeFlowCommand,
        state: DialogueState,
        flows: FlowsList,
    ) -> None:
        target_flow_id = command.flow

        # If the target is already active, nothing to do.
        if state.active_task is not None and state.active_task.flow_id == target_flow_id:
            return

        resumed = state.resume_task(target_flow_id)
        if resumed is None:
            return

        target_flow = flows.get_flow(target_flow_id)
        resumed_name = target_flow.name if target_flow else target_flow_id

        self._activate_system(
            state,
            flows,
            "system_task_resumed",
            ResumedSystemContext(
                resumed_flow_id=target_flow_id,
                resumed_flow_name=resumed_name,
            ),
        )

    # -- helpers -------------------------------------------------------------

    def _get_flow_or_raise(self, flow_id: str, flows: FlowsList) -> Flow:
        flow = flows.get_flow(flow_id)
        if flow is None:
            raise CommandProcessorError(
                f"未知的 flow_id: {flow_id!r}"
            )
        return flow

    def _activate_system(
        self,
        state: DialogueState,
        flows: FlowsList,
        system_flow_id: str,
        system_context: StartedSystemContext
        | InterruptedSystemContext
        | ResumedSystemContext
        | CanceledSystemContext,
    ) -> None:
        """Set the active system flow to *system_flow_id*'s start step."""
        system_flow = flows.get_flow(system_flow_id)
        start_step = system_flow.get_start_step() if system_flow else None
        if start_step is not None:
            system_context.step_id = start_step.id
        state.activate_system_flow(system_context)