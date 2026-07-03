"""Tests for FlowLoader — YAML parsing into FlowsList.

These tests verify that the two real config files (user_flows.yml and
system_flows.yml) are parsed correctly, and that edge cases in the YAML
format are handled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Customer_Service_Assistant.service.task import (
    ActionFlowStep,
    CollectSlotStep,
    ConditionalLink,
    EndFlowStep,
    FallbackLink,
    FlowLoader,
    FlowSlot,
    FlowStepType,
    ResponseDefinition,
    SlotValidation,
    StartFlowStep,
    StaticLink,
)

# ---------------------------------------------------------------------------
# Paths to the real config files
# ---------------------------------------------------------------------------

FLOW_DIR = Path(__file__).resolve().parent.parent / "flow_config"
USER_FLOWS = FLOW_DIR / "user_flows.yml"
SYSTEM_FLOWS = FLOW_DIR / "system_flows.yml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flows_list():
    """Load the real config files once for the module."""
    loader = FlowLoader()
    return loader.load([USER_FLOWS, SYSTEM_FLOWS])


@pytest.fixture(scope="module")
def flow_by_id(flows_list):
    """Return a lookup helper."""

    def _lookup(flow_id: str):
        for f in flows_list.flows:
            if f.id == flow_id:
                return f
        raise KeyError(flow_id)

    return _lookup


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


class TestBasicLoading:
    def test_loads_all_flows(self, flows_list):
        """All 13 flows from the two config files should be present."""
        assert len(flows_list.flows) == 13

    def test_loads_global_slots(self, flows_list):
        """The 8 global slots from user_flows.yml should be loaded."""
        assert len(flows_list.slots) == 8
        assert "order_number" in flows_list.slots
        assert "refund_reason" in flows_list.slots
        assert "tracking_number" in flows_list.slots

    def test_slot_attributes(self, flows_list):
        """Global slots should have their attributes parsed."""
        slot = flows_list.slots["order_number"]
        assert slot.name == "order_number"
        assert slot.type == "text"
        assert slot.label == "订单号"
        assert slot.description == "用户的订单号"


# ---------------------------------------------------------------------------
# Business flows
# ---------------------------------------------------------------------------


class TestBusinessFlows:
    def test_onboarding_has_three_steps(self, flow_by_id):
        flow = flow_by_id("onboarding")
        assert flow.name == "欢迎引导"
        assert len(flow.steps) == 3

    def test_order_status_query_steps(self, flow_by_id):
        flow = flow_by_id("order_status_query")
        assert flow.name == "订单状态查询"
        steps = flow.steps
        assert isinstance(steps[0], StartFlowStep)
        assert isinstance(steps[1], CollectSlotStep)
        assert isinstance(steps[2], ActionFlowStep)  # lookup
        assert isinstance(steps[3], ActionFlowStep)  # show
        assert isinstance(steps[4], EndFlowStep)

    def test_collect_step_response(self, flow_by_id):
        flow = flow_by_id("refund_request")
        collect = flow.steps[1]  # ask_order_number
        assert isinstance(collect, CollectSlotStep)
        assert collect.slot_name == "order_number"
        assert collect.response.mode == "static"
        assert collect.response.text == "请告诉我你的订单号。"

    def test_conditional_next_links(self, flow_by_id):
        flow = flow_by_id("similar_product_recommendation")
        start_step = flow.steps[0]
        assert len(start_step.next) == 2
        assert isinstance(start_step.next[0], ConditionalLink)
        assert start_step.next[0].condition == "slots.get('product_id')"
        assert start_step.next[0].target == "respond"
        assert isinstance(start_step.next[1], FallbackLink)
        assert start_step.next[1].target == "missing_product_context"

    def test_action_step_with_custom_action(self, flow_by_id):
        flow = flow_by_id("logistics_tracking")
        lookup = flow.steps[2]
        assert isinstance(lookup, ActionFlowStep)
        assert lookup.action == "action_lookup_logistics"
        assert lookup.args == {}

    def test_refund_response_contains_slot_placeholders(self, flow_by_id):
        flow = flow_by_id("refund_request")
        respond = flow.steps[3]
        assert isinstance(respond, ActionFlowStep)
        assert "slots.order_number" in respond.args["text"]
        assert "slots.refund_reason" in respond.args["text"]


# ---------------------------------------------------------------------------
# System flows
# ---------------------------------------------------------------------------


class TestSystemFlows:
    def test_all_system_flows_present(self, flow_by_id):
        expected = [
            "system_task_started",
            "system_task_resumed",
            "system_completed",
            "system_cannot_handle",
            "system_collect_information",
            "system_task_interrupted",
            "system_task_canceled",
        ]
        for fid in expected:
            flow_by_id(fid)  # raises KeyError if missing

    def test_collect_information_args_is_string(self, flow_by_id):
        """system_collect_information uses ``args: context.response`` (a string
        reference, not a dict)."""
        flow = flow_by_id("system_collect_information")
        ask = flow.steps[1]
        assert isinstance(ask, ActionFlowStep)
        assert ask.args == "context.response"
        assert isinstance(ask.args, str)

    def test_cannot_handle_has_multiple_conditions(self, flow_by_id):
        flow = flow_by_id("system_cannot_handle")
        start = flow.steps[0]
        assert len(start.next) == 4
        conditions = [
            link for link in start.next if isinstance(link, ConditionalLink)
        ]
        assert len(conditions) == 3
        assert any(
            "clarification_rejected" in c.condition for c in conditions
        )
        fallbacks = [
            link for link in start.next if isinstance(link, FallbackLink)
        ]
        assert len(fallbacks) == 1
        assert fallbacks[0].target == "ask_rephrase"

    def test_interrupted_flow_conditional_next(self, flow_by_id):
        flow = flow_by_id("system_task_interrupted")
        acknowledge = flow.steps[1]
        assert isinstance(acknowledge, ActionFlowStep)
        # acknowledges's next has a conditional + fallback
        assert len(acknowledge.next) == 2
        assert isinstance(acknowledge.next[0], ConditionalLink)
        assert isinstance(acknowledge.next[1], FallbackLink)

    def test_system_completed_is_minimal(self, flow_by_id):
        flow = flow_by_id("system_completed")
        assert len(flow.steps) == 2
        assert isinstance(flow.steps[0], StartFlowStep)
        assert isinstance(flow.steps[1], EndFlowStep)

    def test_system_flows_have_english_names(self, flow_by_id):
        """System flows use English names (not Chinese)."""
        flow = flow_by_id("system_task_started")
        assert flow.name == "task started acknowledgement"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_yaml_does_not_crash(self, tmp_path):
        """A YAML file with no content should produce an empty FlowsList."""
        empty = tmp_path / "empty.yml"
        empty.write_text("", encoding="utf-8")
        loader = FlowLoader()
        result = loader.load([empty])
        assert len(result.flows) == 0
        assert len(result.slots) == 0

    def test_flows_only_section(self, tmp_path):
        """A YAML file with only the flows section (no slots)."""
        f = tmp_path / "test.yml"
        f.write_text(
            """
flows:
  test_flow:
    name: Test
    description: A test flow
    steps:
      - id: start
        type: start
        next: end
      - id: end
        type: end
        next: []
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f])
        assert len(result.flows) == 1
        assert result.flows[0].id == "test_flow"
        assert len(result.flows[0].steps) == 2

    def test_slots_only_section(self, tmp_path):
        """A YAML file with only the slots section (no flows)."""
        f = tmp_path / "test.yml"
        f.write_text(
            """
slots:
  my_slot:
    type: number
    label: My Slot
    description: A test slot
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f])
        assert len(result.flows) == 0
        assert len(result.slots) == 1
        assert result.slots["my_slot"].type == "number"

    def test_merge_slots_across_files(self, tmp_path):
        """Slots from two files should be merged, with first-write wins."""
        f1 = tmp_path / "a.yml"
        f1.write_text(
            """
slots:
  shared:
    type: text
    label: From A
""",
            encoding="utf-8",
        )
        f2 = tmp_path / "b.yml"
        f2.write_text(
            """
slots:
  shared:
    type: number
    label: From B
  extra:
    type: text
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f1, f2])
        # First write wins for "shared"
        assert result.slots["shared"].type == "text"
        assert result.slots["shared"].label == "From A"
        # "extra" is added
        assert result.slots["extra"].type == "text"

    def test_step_with_unknown_type(self, tmp_path):
        """An unknown step type should parse as a generic FlowStep."""
        f = tmp_path / "test.yml"
        f.write_text(
            """
flows:
  test:
    description: Test
    steps:
      - id: mystery
        type: unknown_type
        next: end
      - id: end
        type: end
        next: []
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f])
        step = result.flows[0].steps[0]
        assert step.type == "unknown_type"
        # Should not be a subclass — just the base FlowStep
        assert type(step).__name__ == "FlowStep"

    def test_collect_step_with_validation(self, tmp_path):
        """A collect step with a validation block should parse correctly."""
        f = tmp_path / "test.yml"
        f.write_text(
            """
flows:
  test:
    description: Test
    steps:
      - id: start
        type: start
        next: ask_id
      - id: ask_id
        type: collect
        slot_name: order_number
        response:
          mode: static
          text: "请输入订单号"
        validation:
          condition: "slots.get('order_number')"
          failure_response:
            mode: static
            text: "订单号不能为空"
        next: end
      - id: end
        type: end
        next: []
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f])
        collect = result.flows[0].steps[1]
        assert isinstance(collect, CollectSlotStep)
        assert collect.slot_name == "order_number"
        assert collect.validation is not None
        assert collect.validation.condition == "slots.get('order_number')"
        assert collect.validation.failure_response is not None
        assert collect.validation.failure_response.text == "订单号不能为空"

    def test_action_step_with_rephrase_mode_args(self, tmp_path):
        """An action_response in rephrase mode should preserve text and prompt."""
        f = tmp_path / "test.yml"
        f.write_text(
            """
flows:
  test:
    description: Test
    steps:
      - id: start
        type: start
        next: msg
      - id: msg
        type: action
        action: action_response
        args:
          mode: rephrase
          text: "建议回复"
          prompt: "请改写"
        next: end
      - id: end
        type: end
        next: []
""",
            encoding="utf-8",
        )
        loader = FlowLoader()
        result = loader.load([f])
        action_step = result.flows[0].steps[1]
        assert isinstance(action_step, ActionFlowStep)
        assert action_step.args["mode"] == "rephrase"
        assert action_step.args["text"] == "建议回复"
        assert action_step.args["prompt"] == "请改写"


# ---------------------------------------------------------------------------
# FlowSlot model
# ---------------------------------------------------------------------------


class TestFlowSlot:
    def test_default_values(self):
        slot = FlowSlot(name="test")
        assert slot.name == "test"
        assert slot.type == "any"
        assert slot.label == ""
        assert slot.description == ""


# ---------------------------------------------------------------------------
# ResponseDefinition model
# ---------------------------------------------------------------------------


class TestResponseDefinition:
    def test_default_mode_is_static(self):
        rd = ResponseDefinition()
        assert rd.mode == "static"
        assert rd.text is None
        assert rd.prompt is None


# ---------------------------------------------------------------------------
# SlotValidation model
# ---------------------------------------------------------------------------


class TestSlotValidation:
    def test_defaults_are_none(self):
        sv = SlotValidation()
        assert sv.condition is None
        assert sv.failure_response is None
