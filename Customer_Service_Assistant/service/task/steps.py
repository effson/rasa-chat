"""Flow step model types.

Each step in a flow is one of four concrete types discriminated by
:class:`FlowStepType`:

+------------------+-----------------------------------+
| Type             | Purpose                           |
+------------------+-----------------------------------+
| ``start``        | Flow entry point                  |
+------------------+-----------------------------------+
| ``action``       | Execute a built-in or custom act  |
+------------------+-----------------------------------+
| ``collect``      | Collect a slot value from the user|
+------------------+-----------------------------------+
| ``end``          | Flow terminal point               |
+------------------+-----------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from Customer_Service_Assistant.service.task.links import FlowStepLink


class FlowStepType(str, Enum):
    """Discriminator for the four step kinds."""

    START = "start"
    ACTION = "action"
    COLLECT = "collect"
    END = "end"


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResponseDefinition:
    """Defines how ``action_response`` should produce a reply.

    Attributes:
        mode: One of ``"static"`` (render *text* directly),
              ``"rephrase"`` (rewrite *text* with an LLM using *prompt*),
              or ``"generate"`` (generate from scratch using *prompt*).
        text: The reply template (used in *static* and *rephrase* modes).
        prompt: LLM prompt (used in *rephrase* and *generate* modes).
    """

    mode: str = "static"
    text: str | None = None
    prompt: str | None = None


@dataclass(slots=True)
class SlotValidation:
    """Optional validation applied after collecting a slot value.

    Attributes:
        condition: A Python expression evaluated with ``slots`` in scope.
        failure_response: What to say when *condition* is falsy.
    """

    condition: str | None = None
    failure_response: ResponseDefinition | None = None


# ---------------------------------------------------------------------------
# Step hierarchy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FlowStep:
    """Base class for every step in a flow.

    Attributes:
        id: Unique step identifier within the flow.
        type: Discriminator (``start`` / ``action`` / ``collect`` / ``end``).
        next: Ordered list of outgoing links.
        description: Human-readable explanation of this step's purpose.
    """

    id: str
    type: FlowStepType
    next: List[FlowStepLink] = field(default_factory=list)
    description: str = ""


@dataclass(slots=True)
class StartFlowStep(FlowStep):
    """Flow entry point — always the first step executed.

    Example YAML:

    .. code-block:: yaml

        - id: start
          type: start
          next: ask_order_number
    """

    pass


@dataclass(slots=True)
class ActionFlowStep(FlowStep):
    """Execute a named action (built-in, custom, or control signal).

    *args* is normally a ``dict`` of parameters passed to the action.
    It can also be a ``str`` reference like ``"context.response"`` so that
    the system flow ``system_collect_information`` can forward the collect
    step's own ``response`` block.

    Example YAML:

    .. code-block:: yaml

        - id: respond
          type: action
          action: action_response
          args:
            mode: static
            text: "你好，这里是客服助手。"
          next: end
    """

    action: str = ""
    args: Dict[str, Any] | str = field(default_factory=dict)


@dataclass(slots=True)
class CollectSlotStep(FlowStep):
    """Ask the user for a slot value and wait for their response.

    When the slot already has a value the step is skipped; when it is
    missing the system activates ``system_collect_information`` which
    sends *response* and then issues ``action_listen``.

    Example YAML:

    .. code-block:: yaml

        - id: ask_order_number
          type: collect
          slot_name: order_number
          response:
            mode: static
            text: "请告诉我你的订单号。"
          validation:
            condition: "slots.get('order_number')"
            failure_response:
              mode: static
              text: "订单号不能为空，请重新输入。"
          next: ask_refund_reason
    """

    slot_name: str = ""
    response: ResponseDefinition = field(default_factory=ResponseDefinition)
    validation: SlotValidation | None = None


@dataclass(slots=True)
class EndFlowStep(FlowStep):
    """Flow terminal point — execution of the current flow stops here.

    Example YAML:

    .. code-block:: yaml

        - id: end
          type: end
          next: []
    """

    pass