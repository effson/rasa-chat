"""Top-level flow models — the aggregate object graph produced by FlowLoader.

.. uml::

    FlowsList
      ├── flows: List[Flow]
      │     ├── id, name, description
      │     ├── slots: List[FlowSlot]
      │     └── steps: List[FlowStep]   (→ steps.py)
      └── slots: Dict[str, FlowSlot]    (global slot catalog)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from Customer_Service_Assistant.service.task.steps import FlowStep


@dataclass(slots=True)
class FlowSlot:
    """A named piece of information that a flow needs to collect.

    Attributes:
        name: Machine-readable slot key (e.g. ``"order_number"``).
        type: Expected value type (``"text"``, ``"number"``, ``"any"``, …).
        label: Human-readable label (Chinese, for debug / admin display).
        description: Longer explanation of what this slot represents.
    """

    name: str
    type: str = "any"
    label: str = ""
    description: str = ""


@dataclass(slots=True)
class Flow:
    """A single business or system flow defined by a sequence of steps.

    Attributes:
        id: Unique flow identifier (e.g. ``"refund_request"``,
            ``"system_task_started"``).
        description: What this flow accomplishes.
        steps: Ordered list of steps that make up the flow.
        slots: Flow-specific slot definitions (references into the global
               ``FlowsList.slots`` catalog).
        name: Human-readable flow name (Chinese).
    """

    id: str
    description: str = ""
    steps: List[FlowStep] = field(default_factory=list)
    slots: List[FlowSlot] = field(default_factory=list)
    name: str | None = None


@dataclass(slots=True)
class FlowsList:
    """The complete set of flows loaded from one or more YAML files.

    Attributes:
        flows: Every flow (business + system) keyed by ``flow.id``.
        slots: Global slot catalog shared across all flows.
    """

    flows: List[Flow] = field(default_factory=list)
    slots: Dict[str, FlowSlot] = field(default_factory=dict)