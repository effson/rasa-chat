"""Task handler package — flow definitions, command processing, and flow execution.

Modules
-------
links       — FlowStepLink hierarchy (static, conditional, fallback)
steps       — FlowStep hierarchy (start, action, collect, end) + ResponseDefinition, SlotValidation
models      — Flow, FlowSlot, FlowsList
loader      — FlowLoader: parse YAML config → FlowsList
"""

from Customer_Service_Assistant.service.task.links import (
    ConditionalLink,
    FallbackLink,
    FlowStepLink,
    StaticLink,
)
from Customer_Service_Assistant.service.task.loader import FlowLoader
from Customer_Service_Assistant.service.task.models import Flow, FlowsList, FlowSlot
from Customer_Service_Assistant.service.task.steps import (
    ActionFlowStep,
    CollectSlotStep,
    EndFlowStep,
    FlowStep,
    FlowStepType,
    ResponseDefinition,
    SlotValidation,
    StartFlowStep,
)

__all__ = [
    # links
    "FlowStepLink",
    "StaticLink",
    "ConditionalLink",
    "FallbackLink",
    # steps
    "FlowStepType",
    "ResponseDefinition",
    "SlotValidation",
    "FlowStep",
    "StartFlowStep",
    "ActionFlowStep",
    "CollectSlotStep",
    "EndFlowStep",
    # models
    "FlowSlot",
    "Flow",
    "FlowsList",
    # loader
    "FlowLoader",
]