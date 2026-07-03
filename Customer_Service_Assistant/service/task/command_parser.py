"""Command parser — convert raw dicts (from LLM JSON output) into typed
:class:`Command` objects.

Typical usage::

    from Customer_Service_Assistant.service.task.command_parser import CommandParser

    parser = CommandParser()
    cmd = parser.parse({"command": "start_flow", "flow": "refund_request"})
    # → StartFlowCommand(command="start_flow", flow="refund_request")

Errors are reported via :class:`CommandParseError` so the caller can
decide whether to skip, log, or fall back.
"""

from __future__ import annotations

from typing import Any, Dict

from Customer_Service_Assistant.service.task.commands import (
    _COMMAND_CLASSES,
    CancelFlowCommand,
    Command,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)

# ---------------------------------------------------------------------------
# System flow prefix — start_flow must not target these directly.
# ---------------------------------------------------------------------------
SYSTEM_FLOW_PREFIX = "system_"


class CommandParseError(Exception):
    """Raised when a dict cannot be parsed into a valid Command."""

    pass


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CommandParser:
    """Parse raw dicts into typed :class:`Command` subclasses.

    The parser validates the discriminator, required fields, and
    business rules (e.g. ``start_flow`` must target a business flow,
    not a system flow).
    """

    def parse(self, raw: Dict[str, Any]) -> Command:
        """Convert *raw* into the appropriate :class:`Command` subclass.

        Args:
            raw: A dict with at least a ``"command"`` key.

        Returns:
            A typed command object.

        Raises:
            CommandParseError: If the dict is missing required fields,
                references an unknown command type, or violates a
                business rule.
        """
        if not isinstance(raw, dict):
            raise CommandParseError(
                f"Expected a dict, got {type(raw).__name__}"
            )

        cmd_type = raw.get("command")
        if not cmd_type:
            raise CommandParseError("Missing required field: 'command'")

        cls = _COMMAND_CLASSES.get(cmd_type)
        if cls is None:
            raise CommandParseError(
                f"Unknown command type: {cmd_type!r}. "
                f"Expected one of: {', '.join(sorted(_COMMAND_CLASSES))}"
            )

        # Dispatch to type-specific constructor + validation.
        if cls is StartFlowCommand:
            return self._build_start_flow(raw)
        elif cls is SetSlotsCommand:
            return self._build_set_slots(raw)
        elif cls is CancelFlowCommand:
            return CancelFlowCommand(command=cmd_type)
        elif cls is ResumeFlowCommand:
            return self._build_resume_flow(raw)

        # Should never reach here (all cases are covered), but be safe.
        raise CommandParseError(f"Unhandled command type: {cmd_type!r}")

    # ------------------------------------------------------------------
    # Per-type builders
    # ------------------------------------------------------------------

    def _build_start_flow(self, raw: Dict[str, Any]) -> StartFlowCommand:
        flow = raw.get("flow")
        if not flow:
            raise CommandParseError(
                "start_flow 缺少必填字段 'flow'"
            )
        if not isinstance(flow, str):
            raise CommandParseError(
                f"'flow' 字段必须是字符串，实际类型: {type(flow).__name__}"
            )
        if flow.startswith(SYSTEM_FLOW_PREFIX):
            raise CommandParseError(
                f"不允许直接启动系统 flow: {flow!r}"
            )
        return StartFlowCommand(command="start_flow", flow=flow)

    def _build_set_slots(self, raw: Dict[str, Any]) -> SetSlotsCommand:
        slots = raw.get("slots")
        if slots is None:
            raise CommandParseError(
                "set_slots 缺少必填字段 'slots'"
            )
        if not isinstance(slots, dict):
            raise CommandParseError(
                f"'slots' 字段必须是对象，实际类型: {type(slots).__name__}"
            )
        return SetSlotsCommand(command="set_slots", slots=slots)

    def _build_resume_flow(self, raw: Dict[str, Any]) -> ResumeFlowCommand:
        flow = raw.get("flow")
        if not flow:
            raise CommandParseError(
                "resume_flow 缺少必填字段 'flow'"
            )
        if not isinstance(flow, str):
            raise CommandParseError(
                f"'flow' 字段必须是字符串，实际类型: {type(flow).__name__}"
            )
        return ResumeFlowCommand(command="resume_flow", flow=flow)
