"""Task handler package — flow definitions, command processing, and flow execution.

Modules
-------
links             — FlowStepLink hierarchy (static, conditional, fallback)
steps             — FlowStep hierarchy (start, action, collect, end) + ResponseDefinition, SlotValidation
models            — Flow, FlowSlot, FlowsList
loader            — FlowLoader: parse YAML config → FlowsList
commands          — Command hierarchy (start_flow, set_slots, cancel_flow, resume_flow)
command_parser    — CommandParser: dict/JSON → typed Command objects
command_processor — CommandProcessor: apply Commands to DialogueState
action_runner     — ActionRunner: execute flow actions + built-in action_response / action_listen
flow_executor     — FlowExecutor: advance flows and execute actions until action_listen
task_handler      — TaskHandler: top-level orchestrator (parse → apply → execute → collect)
"""

from Customer_Service_Assistant.service.task.action_runner import (
    ActionRegistry,
    ActionResult,
    ActionRunner,
    create_default_registry,
    CustomAction,
    render_template,
)
from Customer_Service_Assistant.service.task.command_parser import (
    CommandParseError,
    CommandParser,
)
from Customer_Service_Assistant.service.task.command_processor import (
    CommandProcessor,
    CommandProcessorError,
)
from Customer_Service_Assistant.service.task.commands import (
    CancelFlowCommand,
    Command,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)
from Customer_Service_Assistant.service.task.flow_executor import FlowExecutor
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
from Customer_Service_Assistant.service.task.task_handler import (
    TaskHandler,
    TaskHandlerError,
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
    # commands
    "Command",
    "StartFlowCommand",
    "SetSlotsCommand",
    "CancelFlowCommand",
    "ResumeFlowCommand",
    # command_parser
    "CommandParser",
    "CommandParseError",
    # command_processor
    "CommandProcessor",
    "CommandProcessorError",
    # action_runner
    "ActionRunner",
    "ActionResult",
    "ActionRegistry",
    "CustomAction",
    "create_default_registry",
    "render_template",
    # flow_executor
    "FlowExecutor",
    # task_handler
    "TaskHandler",
    "TaskHandlerError",
]