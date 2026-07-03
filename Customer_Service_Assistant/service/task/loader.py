"""Flow loader — parse YAML flow configuration files into a :class:`FlowsList`.

Typical usage::

    from pathlib import Path
    from Customer_Service_Assistant.service.task.loader import FlowLoader

    loader = FlowLoader()
    flows = loader.load([
        Path("flow_config/user_flows.yml"),
        Path("flow_config/system_flows.yml"),
    ])

    refund = [f for f in flows.flows if f.id == "refund_request"][0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from Customer_Service_Assistant.service.task.links import (
    ConditionalLink,
    FallbackLink,
    FlowStepLink,
    StaticLink,
)
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

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FlowLoader:
    """Parse one or more YAML flow-configuration files into a ``FlowsList``."""

    def load(self, file_paths: List[str | Path]) -> FlowsList:
        """Load and merge flows from *file_paths*.

        Args:
            file_paths: Paths to ``.yml`` files, each containing a top-level
                ``flows:`` section and an optional ``slots:`` section.

        Returns:
            A fully parsed ``FlowsList`` with all flows and global slots.
        """
        all_flows: List[Flow] = []
        all_slots: Dict[str, FlowSlot] = {}

        for file_path in file_paths:
            file_path = Path(file_path)
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)

            if data is None:
                continue

            # -- Global slot catalog ------------------------------------------
            raw_slots: Dict[str, Any] = data.get("slots") or {}
            for slot_name, slot_data in raw_slots.items():
                if slot_name not in all_slots:
                    all_slots[slot_name] = FlowSlot(
                        name=slot_name,
                        type=slot_data.get("type", "any"),
                        label=slot_data.get("label", ""),
                        description=slot_data.get("description", ""),
                    )

            # -- Flows ---------------------------------------------------------
            raw_flows: Dict[str, Any] = data.get("flows") or {}
            for flow_id, flow_data in raw_flows.items():
                raw_steps: List[Dict[str, Any]] = flow_data.get("steps") or []
                steps = [self._parse_step(raw_step) for raw_step in raw_steps]

                # Flow-specific slot references
                flow_slot_names: List[str] = flow_data.get("slots") or []
                flow_slots: List[FlowSlot] = []
                for slot_name in flow_slot_names:
                    if slot_name in all_slots:
                        flow_slots.append(all_slots[slot_name])
                    else:
                        flow_slots.append(FlowSlot(name=slot_name))

                flow = Flow(
                    id=flow_id,
                    name=flow_data.get("name"),
                    description=flow_data.get("description", ""),
                    steps=steps,
                    slots=flow_slots,
                )
                all_flows.append(flow)

        return FlowsList(flows=all_flows, slots=all_slots)

    # ------------------------------------------------------------------
    # Step parsing
    # ------------------------------------------------------------------

    def _parse_step(self, raw: Dict[str, Any]) -> FlowStep:
        """Parse a single step dict into the correct ``FlowStep`` subclass."""
        type_str = raw.get("type", "")
        try:
            step_type = FlowStepType(type_str)
        except ValueError:
            # Unknown step type — return a generic FlowStep so loading
            # doesn't break; the executor will skip steps it can't handle.
            return FlowStep(
                id=raw.get("id", ""),
                type=type_str,  # type: ignore[arg-type]
                next=self._parse_next(raw.get("next")),
                description=raw.get("description", ""),
            )

        common: Dict[str, Any] = dict(
            id=raw.get("id", ""),
            type=step_type,
            next=self._parse_next(raw.get("next")),
            description=raw.get("description", ""),
        )

        if step_type == FlowStepType.START:
            return StartFlowStep(**common)

        elif step_type == FlowStepType.END:
            return EndFlowStep(**common)

        elif step_type == FlowStepType.ACTION:
            action = raw.get("action", "")
            args = raw.get("args") or {}
            # args can be a dict of parameters OR a string reference like
            # "context.response" (used by system_collect_information).
            return ActionFlowStep(action=action, args=args, **common)

        elif step_type == FlowStepType.COLLECT:
            slot_name = raw.get("slot_name", "")
            response = self._parse_response(raw.get("response"))
            validation = self._parse_validation(raw.get("validation"))
            return CollectSlotStep(
                slot_name=slot_name,
                response=response,
                validation=validation,
                **common,
            )

        # Unknown step type — return a generic FlowStep so loading doesn't
        # break; the executor will skip steps it cannot handle.
        return FlowStep(**common)

    # ------------------------------------------------------------------
    # Next-link parsing
    # ------------------------------------------------------------------

    def _parse_next(self, raw_next: object) -> List[FlowStepLink]:
        """Convert a YAML *next* field into an ordered list of links.

        ==================== ===========================================
        YAML shape            Result
        ==================== ===========================================
        ``"step_id"``         ``[StaticLink(target="step_id")]``
        ``[{if:, then:}, …]`` ``[ConditionalLink(…), …, FallbackLink(…)]``
        ``None`` / ``[]``     ``[]``
        ==================== ===========================================
        """
        if raw_next is None:
            return []

        if isinstance(raw_next, str):
            return [StaticLink(target=raw_next)]

        if isinstance(raw_next, list):
            links: List[FlowStepLink] = []
            for entry in raw_next:
                if not isinstance(entry, dict):
                    continue
                if "if" in entry:
                    links.append(
                        ConditionalLink(
                            condition=str(entry["if"]),
                            target=str(entry.get("then", "")),
                        )
                    )
                elif "else" in entry:
                    links.append(
                        FallbackLink(target=str(entry.get("else", "")))
                    )
            return links

        return []

    # ------------------------------------------------------------------
    # Response / validation helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: Dict[str, Any] | None
    ) -> ResponseDefinition:
        """Build a ``ResponseDefinition`` from a YAML mapping (or ``None``)."""
        if raw is None:
            return ResponseDefinition()
        return ResponseDefinition(
            mode=raw.get("mode", "static"),
            text=raw.get("text"),
            prompt=raw.get("prompt"),
        )

    def _parse_validation(
        self, raw: Dict[str, Any] | None
    ) -> SlotValidation | None:
        """Build a ``SlotValidation`` from a YAML mapping (or ``None``)."""
        if raw is None:
            return None
        failure_response = (
            self._parse_response(raw["failure_response"])
            if raw.get("failure_response")
            else None
        )
        return SlotValidation(
            condition=raw.get("condition"),
            failure_response=failure_response,
        )