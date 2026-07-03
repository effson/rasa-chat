"""Command models — structured task-state operations extracted from the
TurnPlanner's output.

Every turn the planner produces a :class:`TurnPlan`.  When the plan's
*direction* is ``"task"`` and the plan passes validation, it is translated
into one or more :class:`Command` objects that describe **what should happen
to the task state** before flow execution resumes.

Command types
-------------
+----------------------+---------------------------------------------+
| Command              | Trigger                                     |
+----------------------+---------------------------------------------+
| ``start_flow``       | User wants to begin (or switch to) a flow   |
+----------------------+---------------------------------------------+
| ``set_slots``        | User provided slot values for the active    |
|                      | task (extracted from this turn's message)   |
+----------------------+---------------------------------------------+
| ``cancel_flow``      | User wants to abandon the active task       |
+----------------------+---------------------------------------------+
| ``resume_flow``      | User wants to resume a previously paused    |
|                      | task                                        |
+----------------------+---------------------------------------------+

JSON shapes (what the planner / upstream produces)
--------------------------------------------------

*start_flow*:

.. code-block:: json

    {"command": "start_flow", "flow": "refund_request"}

*set_slots*:

.. code-block:: json

    {"command": "set_slots", "slots": {"order_number": "10001"}}

*cancel_flow*:

.. code-block:: json

    {"command": "cancel_flow"}

*resume_flow*:

.. code-block:: json

    {"command": "resume_flow", "flow": "refund_request"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class Command:
    """Base class for all task-state commands.

    Attributes:
        command: The command discriminator string (``"start_flow"``,
                 ``"set_slots"``, ``"cancel_flow"``, ``"resume_flow"``).
    """

    command: str


@dataclass(slots=True)
class StartFlowCommand(Command):
    """Start (or switch to) a business flow.

    *flow* must be a business flow id — system flows (those starting with
    ``system_``) cannot be started directly by the user.
    """

    flow: str


@dataclass(slots=True)
class SetSlotsCommand(Command):
    """Write one or more slot values into the active task.

    The upstream planner extracts slot-like information from the user's
    message (e.g. an order number the user typed) and packages it as a
    ``set_slots`` command.
    """

    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CancelFlowCommand(Command):
    """Cancel the currently active task."""

    pass


@dataclass(slots=True)
class ResumeFlowCommand(Command):
    """Resume a previously paused task.

    *flow* identifies which paused task to bring back as the active task.
    """

    flow: str


# ---------------------------------------------------------------------------
# Discriminator map — used by CommandParser for O(1) lookup
# ---------------------------------------------------------------------------

_COMMAND_CLASSES: Dict[str, type[Command]] = {
    "start_flow": StartFlowCommand,
    "set_slots": SetSlotsCommand,
    "cancel_flow": CancelFlowCommand,
    "resume_flow": ResumeFlowCommand,
}
