"""Tests for Command models and CommandParser."""

from __future__ import annotations

import pytest

from Customer_Service_Assistant.service.task import (
    CancelFlowCommand,
    Command,
    CommandParseError,
    CommandParser,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parser():
    return CommandParser()


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------


class TestCommandModels:
    def test_command_base(self):
        cmd = Command(command="unknown")
        assert cmd.command == "unknown"

    def test_start_flow_command(self):
        cmd = StartFlowCommand(command="start_flow", flow="refund_request")
        assert cmd.command == "start_flow"
        assert cmd.flow == "refund_request"
        assert isinstance(cmd, Command)

    def test_set_slots_command(self):
        cmd = SetSlotsCommand(
            command="set_slots",
            slots={"order_number": "10001"},
        )
        assert cmd.command == "set_slots"
        assert cmd.slots == {"order_number": "10001"}
        assert isinstance(cmd, Command)

    def test_set_slots_defaults_to_empty_dict(self):
        cmd = SetSlotsCommand(command="set_slots")
        assert cmd.slots == {}

    def test_cancel_flow_command(self):
        cmd = CancelFlowCommand(command="cancel_flow")
        assert cmd.command == "cancel_flow"
        assert isinstance(cmd, Command)

    def test_resume_flow_command(self):
        cmd = ResumeFlowCommand(command="resume_flow", flow="refund_request")
        assert cmd.command == "resume_flow"
        assert cmd.flow == "refund_request"
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# Parsing — happy paths
# ---------------------------------------------------------------------------


class TestParseHappyPath:
    def test_parse_start_flow(self, parser):
        cmd = parser.parse({"command": "start_flow", "flow": "refund_request"})
        assert isinstance(cmd, StartFlowCommand)
        assert cmd.command == "start_flow"
        assert cmd.flow == "refund_request"

    def test_parse_set_slots_single(self, parser):
        cmd = parser.parse(
            {"command": "set_slots", "slots": {"order_number": "10001"}}
        )
        assert isinstance(cmd, SetSlotsCommand)
        assert cmd.slots == {"order_number": "10001"}

    def test_parse_set_slots_multiple(self, parser):
        cmd = parser.parse(
            {
                "command": "set_slots",
                "slots": {
                    "order_number": "10086",
                    "refund_reason": "商品有破损",
                },
            }
        )
        assert isinstance(cmd, SetSlotsCommand)
        assert len(cmd.slots) == 2

    def test_parse_set_slots_empty(self, parser):
        cmd = parser.parse({"command": "set_slots", "slots": {}})
        assert isinstance(cmd, SetSlotsCommand)
        assert cmd.slots == {}

    def test_parse_cancel_flow(self, parser):
        cmd = parser.parse({"command": "cancel_flow"})
        assert isinstance(cmd, CancelFlowCommand)

    def test_parse_resume_flow(self, parser):
        cmd = parser.parse(
            {"command": "resume_flow", "flow": "order_status_query"}
        )
        assert isinstance(cmd, ResumeFlowCommand)
        assert cmd.flow == "order_status_query"


# ---------------------------------------------------------------------------
# Parsing — error paths
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_non_dict_input(self, parser):
        with pytest.raises(CommandParseError, match="Expected a dict"):
            parser.parse("not a dict")  # type: ignore[arg-type]

    def test_missing_command_field(self, parser):
        with pytest.raises(CommandParseError, match="Missing required field"):
            parser.parse({"flow": "refund_request"})

    def test_unknown_command_type(self, parser):
        with pytest.raises(CommandParseError, match="Unknown command type"):
            parser.parse({"command": "do_something_else"})

    def test_empty_command_field(self, parser):
        with pytest.raises(CommandParseError, match="Missing required field"):
            parser.parse({"command": ""})

    def test_start_flow_missing_flow_field(self, parser):
        with pytest.raises(CommandParseError, match="缺少必填字段"):
            parser.parse({"command": "start_flow"})

    def test_start_flow_empty_flow(self, parser):
        with pytest.raises(CommandParseError, match="缺少必填字段"):
            parser.parse({"command": "start_flow", "flow": ""})

    def test_start_flow_flow_not_string(self, parser):
        with pytest.raises(CommandParseError, match="'flow' 字段必须是字符串"):
            parser.parse({"command": "start_flow", "flow": 123})

    def test_set_slots_missing_slots_field(self, parser):
        with pytest.raises(CommandParseError, match="缺少必填字段"):
            parser.parse({"command": "set_slots"})

    def test_set_slots_slots_not_dict(self, parser):
        with pytest.raises(CommandParseError, match="'slots' 字段必须是对象"):
            parser.parse({"command": "set_slots", "slots": "not_a_dict"})

    def test_resume_flow_missing_flow_field(self, parser):
        with pytest.raises(CommandParseError, match="缺少必填字段"):
            parser.parse({"command": "resume_flow"})

    def test_resume_flow_empty_flow(self, parser):
        with pytest.raises(CommandParseError, match="缺少必填字段"):
            parser.parse({"command": "resume_flow", "flow": ""})


# ---------------------------------------------------------------------------
# Business rule: no system flow via start_flow
# ---------------------------------------------------------------------------


class TestSystemFlowGuard:
    def test_start_flow_rejects_system_collect_information(self, parser):
        with pytest.raises(CommandParseError, match="不允许直接启动系统 flow"):
            parser.parse(
                {"command": "start_flow", "flow": "system_collect_information"}
            )

    def test_start_flow_rejects_system_task_started(self, parser):
        with pytest.raises(CommandParseError, match="不允许直接启动系统 flow"):
            parser.parse(
                {"command": "start_flow", "flow": "system_task_started"}
            )

    def test_start_flow_rejects_any_system_prefix(self, parser):
        with pytest.raises(CommandParseError, match="不允许直接启动系统 flow"):
            parser.parse({"command": "start_flow", "flow": "system_anything"})

    def test_resume_flow_allows_system_flow(self, parser):
        """resume_flow does not block system flows — that's a concern for
        CommandProcessor which has access to actual paused task state."""
        cmd = parser.parse(
            {"command": "resume_flow", "flow": "system_something"}
        )
        assert isinstance(cmd, ResumeFlowCommand)


# ---------------------------------------------------------------------------
# Round-trip: parse then inspect
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_start_flow_round_trip(self, parser):
        cmd = parser.parse({"command": "start_flow", "flow": "refund_request"})
        assert isinstance(cmd, StartFlowCommand)
        assert cmd.command == "start_flow"
        assert cmd.flow == "refund_request"

    def test_set_slots_round_trip(self, parser):
        slots = {"order_number": "10086", "reason": "破损"}
        cmd = parser.parse({"command": "set_slots", "slots": slots})
        assert isinstance(cmd, SetSlotsCommand)
        assert cmd.slots == slots

    def test_cancel_flow_round_trip(self, parser):
        cmd = parser.parse({"command": "cancel_flow"})
        assert isinstance(cmd, CancelFlowCommand)
        assert cmd.command == "cancel_flow"

    def test_list_parsing_typical_turn_plan_commands(self, parser):
        """Simulate a batch of commands that might come from a single turn."""
        raw_commands = [
            {"command": "start_flow", "flow": "refund_request"},
            {"command": "set_slots", "slots": {"order_number": "10086"}},
        ]
        cmds = [parser.parse(r) for r in raw_commands]
        assert isinstance(cmds[0], StartFlowCommand)
        assert isinstance(cmds[1], SetSlotsCommand)
        assert cmds[0].flow == "refund_request"
        assert cmds[1].slots == {"order_number": "10086"}