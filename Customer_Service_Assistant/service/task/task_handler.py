"""TaskHandler — top-level orchestrator for task-oriented dialogue turns.

This is step 6 of the TaskHandler pipeline (§1 of the detailed design doc).
TaskHandler receives pre-parsed command dicts (from TurnPlanner via the
engine) and the current :class:`~Customer_Service_Assistant.service.schemas.DialogueState`,
applies commands to update task state, advances flows to produce bot
messages, and returns the collected messages so the engine can attach
them to ``pending_turn.assistant_messages``.

Internal pipeline (per §4 of the design doc):

.. code-block:: text

    commands → CommandParser → CommandProcessor → FlowExecutor → messages

Typical usage::

    from pathlib import Path
    from Customer_Service_Assistant.service.task import (
        CommandParser, CommandProcessor, FlowExecutor, ActionRunner, FlowLoader,
    )
    from Customer_Service_Assistant.service.task.task_handler import TaskHandler

    flows = FlowLoader().load([
        Path("flow_config/user_flows.yml"),
        Path("flow_config/system_flows.yml"),
    ])
    runner = ActionRunner()
    handler = TaskHandler(
        command_processor=CommandProcessor(),
        command_parser=CommandParser(),
        flow_executor=FlowExecutor(runner),
        flows=flows,
    )
    messages = await handler.run(
        commands=[{"command": "start_flow", "flow": "refund_request"}],
        state=state,
    )
"""

from __future__ import annotations

from typing import List

from Customer_Service_Assistant.service.schemas import DialogueState, Message
from Customer_Service_Assistant.service.task.command_parser import (
    CommandParseError,
    CommandParser,
)
from Customer_Service_Assistant.service.task.command_processor import (
    CommandProcessor,
    CommandProcessorError,
)
from Customer_Service_Assistant.service.task.commands import Command
from Customer_Service_Assistant.service.task.flow_executor import FlowExecutor
from Customer_Service_Assistant.service.task.models import FlowsList

import logging

logger = logging.getLogger(__name__)


class TaskHandlerError(Exception):
    """Raised when the TaskHandler cannot process a turn."""

    pass


class TaskHandler:
    """Orchestrate a single task-oriented dialogue turn.

    Parameters
    ----------
    command_processor:
        Applies typed ``Command`` objects to ``DialogueState``.
    command_parser:
        Parses raw dicts (from LLM planner output) into ``Command`` objects.
    flow_executor:
        Advances flows and executes actions until ``action_listen``.
    flows:
        The complete set of business and system flows.
    """

    def __init__(
        self,
        command_processor: CommandProcessor,
        command_parser: CommandParser,
        flow_executor: FlowExecutor,
        flows: FlowsList,
    ) -> None:
        self._command_processor = command_processor
        self._command_parser = command_parser
        self._flow_executor = flow_executor
        self._flows = flows

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        commands: List[dict],
        state: DialogueState,
    ) -> List[Message]:
        """Process *commands* and execute flows for this turn.

        Pipeline
        --------
        1. Parse raw command dicts into typed :class:`Command` objects.
           Malformed commands are logged and skipped (the turn still
           runs with whatever commands parsed successfully).
        2. Apply parsed commands to *state* via :class:`CommandProcessor`.
           This updates ``active_task``, ``paused_tasks``, and
           ``active_system_flow``.
        3. Execute flows via :class:`FlowExecutor`, which advances
           through system / business flows and executes actions until
           ``action_listen`` is hit (or no work remains).
        4. Return the collected bot reply messages.

        Returns
        -------
        list[Message]
            Bot reply messages produced during flow execution.  The
            caller (typically :class:`DialogueEngine`) attaches these
            to ``pending_turn.assistant_messages``.
        """
        # Step 1 — parse commands (best-effort; skip malformed ones)
        parsed_commands: List[Command] = []
        for raw in commands:
            try:
                parsed_commands.append(self._command_parser.parse(raw))
            except CommandParseError as exc:
                logger.warning(
                    "TaskHandler: skipping malformed command %r — %s",
                    raw,
                    exc,
                )
                continue

        # Step 2 — apply commands to state
        if parsed_commands:
            try:
                self._command_processor.run(parsed_commands, state, self._flows)
            except CommandProcessorError as exc:
                logger.error(
                    "TaskHandler: CommandProcessor failed — %s",
                    exc,
                )
                raise TaskHandlerError(
                    f"命令处理失败: {exc}"
                ) from exc

        # Step 3 — execute flows (collecting bot messages)
        messages = await self._flow_executor.run(state, self._flows)

        # Step 4 — return messages
        return messages