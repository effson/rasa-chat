"""Engine layer — core dispatcher that orchestrates a single turn.

Flow:  Prepare Session → Create Turn → Classify Message →
       (Object | Text) → Planning → Validate → Route → Generate → Commit

Steps 1-5 were implemented first; step 6 (validation) is added here.
"""

from __future__ import annotations

import json
import re
import time

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.chitchat_handler import ChitChatHandler
from Customer_Service_Assistant.service.schemas import (
    DialogueState,
    Message,
    Turn,
    TurnPlan,
    ValidationResult,
)
from Customer_Service_Assistant.service.validator import TurnPlanValidator

# ---------------------------------------------------------------------------
# Session timeout (seconds).  If the user's last activity was longer ago than
# this, the current session is considered expired and the engine starts a
# fresh session after clearing all running state.
# ---------------------------------------------------------------------------
SESSION_TIMEOUT = 30 * 60  # 30 minutes


class DialogueEngine:
    """Core dispatcher for one conversation turn.

    Coordinates Planning (prompt building / validation), Routing (track
    selection), and delegates generation to the appropriate track.  When
    Planning and Route layers are built out they will be injected here;
    for now the engine handles everything inline.
    """

    def __init__(self) -> None:
        self._llm = llm
        self._validator = TurnPlanValidator()
        self._chitchat = ChitChatHandler()

    # -- public API ----------------------------------------------------------

    async def run(self, state: DialogueState, user_message: Message) -> Message:
        """Process *state* with the incoming *user_message* and return the
        bot's reply as a ``Message``.

        Steps (per DialogueEngine 设计.md):

        1. Prepare Session
        2. Create Turn
        3. Classify message — text, object, or both
        4. Process object — write focused_object, check slot match
        5. Process text — TurnPlanner → structured TurnPlan
        6. Validate plan — TurnPlanValidator → route or clarify
        """
        # 1. Prepare Session
        self._prepare_session(state)

        # 2. Create Turn
        turn = Turn(input_message=user_message)
        state.pending_turn = turn

        # 3. Classify & dispatch
        has_text = user_message.text is not None
        has_object = user_message.object is not None

        if has_object:
            # Step 4 — object messages: write to focused_object, check slots
            self._process_object_message(state, user_message)

        if has_text:
            # Steps 5-6 — plan → validate → route → generate
            plan = await self._plan(state, user_message.text)

            result = self._validate_plan(plan, state, user_message.text)

            if not result.is_valid:
                # Plan failed validation — generate clarification
                bot_msg = await self._generate_clarification(
                    plan, result, state, user_message.text
                )
            else:
                # Plan is valid — route to the appropriate handler
                bot_msg = await self._generate_for_plan(
                    plan, result, state
                )
        else:
            # Pure object message (no text) — respond based on slot match
            bot_msg = self._respond_to_object(state)

        # Store bot reply in the pending turn
        turn.assistant_messages.append(bot_msg)
        return bot_msg

    # -- Session preparation -------------------------------------------------

    def _prepare_session(self, state: DialogueState) -> None:
        """Ensure the current session is still valid (step 1 in the design doc).

        * If there is **no** current session, create one.
        * If the current session **has not** timed out, keep it.
        * If the current session **has** timed out:

          1. Mark the old session as closed (``closed_at``).
          2. Clear all running state — ``active_task``, ``paused_tasks``,
             ``active_system_flow``, ``focused_object``.
          3. Create a fresh session.
        """
        now = time.time()
        session = state.current_session

        if session is not None:
            elapsed = now - session.last_activity_at
            if elapsed <= SESSION_TIMEOUT:
                return
            # Session timed out — close it and clear running state
            session.closed_at = now
            state.current_session_id = None
            state.active_task = None
            state.paused_tasks = []
            state.active_system_flow = None
            state.focused_object = None

        # Create a fresh session (handles both first-time and post-timeout)
        state.ensure_session()

    # -- Step 4: process object messages ------------------------------------

    def _process_object_message(
        self, state: DialogueState, user_message: Message,
    ) -> None:
        """Write the object from *user_message* into ``state.focused_object``
        and check whether it fills a slot the current task is waiting for.

        Design doc §2.4 — processing object messages.
        """
        obj = user_message.object

        # Always update focused_object so subsequent turns have context
        state.focused_object = obj

        # Check if this object matches a slot the active task is collecting.
        # When TaskHandler is built, this becomes a proper slot-fill check;
        # for now we surface the state so text-side planning can use it.
        active = state.active_task
        if active is not None:
            # Mark the object type in the task slots so TurnPlanner can
            # discover that the user just provided this piece of information.
            active.slots[obj.type] = {"id": obj.id, "title": obj.title}

    def _respond_to_object(self, state: DialogueState) -> Message:
        """Generate a response for a pure object message (no accompanying text).

        If the active task's slots were just filled by this object, the task
        will be resumed by TaskHandler (future).  Form now we give a brief
        acknowledgement that prompts the user to say what they need.
        """
        obj = state.focused_object
        label = (obj.title or obj.id) if obj else "该对象"

        # When TaskHandler is built, this branch routes through
        # ClarifyResponder or TaskHandler depending on slot matching.
        return Message(
            role="bot",
            text=f"已收到你选择的{label}，请问需要我帮你做什么？",
        )

    # -- Step 5: process text messages (TurnPlanner) --------------------------

    def _build_plan_context(self, state: DialogueState) -> str:
        """Build the system-context portion of the planning prompt.

        This is the prompt-construction logic that was previously in
        ``_plan()``, extracted so the LLM-calling ``_plan()`` method
        stays focused on the structured-output interaction.
        """
        parts: list[str] = [
            "你是一个电商客服助手，负责理解用户意图并输出结构化计划。",
        ]

        # -- Context: active task --------------------------------------------
        active = state.active_task
        if active is not None:
            parts.append(
                f"当前正在处理：{active.flow_id}（步骤 {active.step_id}）。"
                f"用户可能在继续此任务。"
            )

        # -- Context: paused tasks -------------------------------------------
        paused = state.paused_tasks
        if paused:
            names = ", ".join(p.flow_id for p in paused)
            parts.append(
                f"用户有暂停的任务：{names}。用户可能想回到其中之一。"
            )

        # -- Context: focused object -----------------------------------------
        focus = state.focused_object
        if focus is not None:
            label = focus.title or focus.id
            parts.append(f"用户当前关注的对象：[{focus.type}] {label}。")

        return "\n".join(parts)

    async def _plan(self, state: DialogueState, user_text: str) -> TurnPlan:
        """Call the LLM to produce a structured ``TurnPlan``.

        Design doc §2.5 — TurnPlanner considers recent dialogue, active
        tasks, paused tasks, focused objects, and available capabilities.
        """

        system_context = self._build_plan_context(state)

        # -- Planning instruction with JSON schema ---------------------------
        planning_instruction = f"""{system_context}

你可以处理以下业务：查订单、查物流、申请退款、催发货、推荐商品。
你可以回答以下知识类问题：商品信息、退换货政策、常见问题。

请分析用户最后一条消息的意图，输出一个 JSON 计划。

输出格式（严格 JSON，不要输出其他内容）：
{{
    "direction": "task" | "knowledge" | "chitchat" | "invalid",
    "reason": "一句话说明判断依据",
    "flow_id": "业务流ID（direction=task 时必填，可选值：order_status_query, logistics_tracking, refund_request, similar_product_recommendation, human_handoff, onboarding）",
    "action": "start | resume | cancel | continue（direction=task 时填写）",
    "knowledge_intent": "知识意图（direction=knowledge 时必填，可选值：商品信息, 退换货政策, 常见问题）",
    "missing_info": "缺少哪些关键信息（direction=invalid 时填写）",
    "conflicts": ["方向冲突1", "方向冲突2"]
}}

规则：
- 如果无法确定用户意图，direction 设为 "invalid"。
- 如果用户消息中有"它"、"这个"但无法确定具体指什么对象，direction 设为 "invalid"。
- 如果用户只是打招呼、闲聊、感谢，direction 设为 "chitchat"。
- 每轮只能选一个主方向。
"""

        # -- Conversation history --------------------------------------------
        messages: list[dict] = [
            {"role": "system", "content": planning_instruction}
        ]

        for msg in state.current_messages:
            role = "user" if msg.role == "user" else "assistant"
            content_parts: list[str] = []
            if msg.text:
                content_parts.append(msg.text)
            if msg.object:
                content_parts.append(
                    f"[{msg.object.type}: {msg.object.title or msg.object.id}]"
                )
            messages.append({"role": role, "content": "\n".join(content_parts)})

        # Append the current user message (may already be in current_messages
        # via pending_turn, but be explicit so the planner always sees it).
        messages.append({"role": "user", "content": user_text})

        # -- LLM call --------------------------------------------------------
        response = await self._llm.ainvoke(messages)
        raw = response.content.strip() if response.content else ""

        return self._parse_plan_json(raw)

    def _parse_plan_json(self, raw: str) -> TurnPlan:
        """Parse the LLM's JSON response into a ``TurnPlan``.

        Handles common LLM output quirks: markdown code fences, trailing
        commas, and stray text before / after the JSON object.
        """
        # Strip markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # Find the outermost JSON object
        match = re.search(r"\{.*}", raw, re.DOTALL)
        if not match:
            # Can't find any JSON — fall back to an invalid plan
            return TurnPlan(
                direction="invalid",
                reason=f"无法解析 LLM 输出为 JSON: {raw[:200]}",
                missing_info="LLM 输出格式错误",
            )

        json_str = match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return TurnPlan(
                direction="invalid",
                reason=f"JSON 解析失败: {json_str[:200]}",
                missing_info="LLM 输出 JSON 格式错误",
            )

        # Coerce the raw dict into a TurnPlan, filling in defaults for
        # any missing or malformed fields.
        try:
            return TurnPlan(
                direction=self._coerce_direction(data.get("direction")),
                reason=str(data.get("reason", "")),
                flow_id=data.get("flow_id") if data.get("flow_id") else None,
                action=self._coerce_action(data.get("action")),
                knowledge_intent=(
                    data.get("knowledge_intent")
                    if data.get("knowledge_intent")
                    else None
                ),
                missing_info=(
                    data.get("missing_info")
                    if data.get("missing_info")
                    else None
                ),
                conflicts=(
                    data.get("conflicts")
                    if isinstance(data.get("conflicts"), list)
                    else []
                ),
            )
        except Exception:
            return TurnPlan(
                direction="invalid",
                reason=f"TurnPlan 构造失败: {json_str[:200]}",
                missing_info="LLM 输出结构不完整",
            )

    @staticmethod
    def _coerce_direction(raw: object) -> str:
        """Normalize the direction field, defaulting to 'invalid'."""
        if isinstance(raw, str) and raw in ("task", "knowledge", "chitchat"):
            return raw
        return "invalid"

    @staticmethod
    def _coerce_action(raw: object) -> str | None:
        """Normalize the action field, defaulting to None."""
        if isinstance(raw, str) and raw in (
            "start", "resume", "cancel", "continue",
        ):
            return raw
        return None

    # -- Step 6: validate plan ------------------------------------------------

    def _validate_plan(
        self,
        plan: TurnPlan,
        state: DialogueState,
        user_text: str,
    ) -> ValidationResult:
        """Run ``TurnPlanValidator`` on the planner's output.

        Design doc §2.6 — the plan must be checked before execution.
        """
        return self._validator.validate(plan, state, user_text)

    # -- Generate: clarification (plan invalid) -------------------------------

    async def _generate_clarification(
        self,
        _plan: TurnPlan,  # reserved for ClarifyResponder (step 7)
        result: ValidationResult,
        state: DialogueState,
        user_text: str,
    ) -> Message:
        """Generate a clarification question when the plan fails validation.

        Design doc §2.7 — ClarifyResponder.  When step 7 is fully
        implemented, this body will delegate to a dedicated ClarifyResponder
        class.  For now the engine generates a targeted follow-up inline.
        """
        issues_text = "；".join(result.issues)

        system_parts: list[str] = [
            "你是一个电商客服助手。用户的意图不明确，你需要友好地追问以澄清。",
            "追问要自然、简短，一次只问一个最关键的问题。",
            f"已知问题：{issues_text}",
        ]

        focus = state.focused_object
        if focus is not None:
            label = focus.title or focus.id
            system_parts.append(
                f"用户当前关注 [{focus.type}] {label}，可以围绕这个对象追问。"
            )

        messages: list[dict] = [
            {"role": "system", "content": "\n".join(system_parts)}
        ]

        for msg in state.current_messages:
            role = "user" if msg.role == "user" else "assistant"
            if msg.text:
                messages.append({"role": role, "content": msg.text})

        messages.append({"role": "user", "content": user_text})

        response = await self._llm.ainvoke(messages)
        text = response.content.strip() if response.content else "抱歉，我没有完全理解你的意思，可以再说具体一点吗？"

        return Message(role="bot", text=text)

    # -- Generate: routed (plan valid) ----------------------------------------

    async def _generate_for_plan(
        self,
        _plan: TurnPlan,  # reserved for TaskHandler et al. (steps 8-10)
        result: ValidationResult,
        state: DialogueState,
    ) -> Message:
        """Dispatch to the appropriate handler based on validated direction.

        - chitchat → ChitChatHandler (implemented)
        - task    → inline prompt (placeholder for TaskHandler, step 8)
        - knowledge → inline prompt (placeholder for KnowledgeHandler, step 9)
        """
        direction = result.direction

        if direction == "chitchat":
            # The pending turn's input message is the user text for this turn.
            user_text = ""
            if state.pending_turn and state.pending_turn.input_message.text:
                user_text = state.pending_turn.input_message.text
            return await self._chitchat.handle(state, user_text)

        # -- Placeholder for task / knowledge (will be replaced by dedicated
        #    handlers when steps 8-9 are implemented) -------------------------

        direction_hints: dict[str, str] = {
            "task": (
                "你正在帮用户办理业务。请根据对话上下文生成有用的回复。"
                "如果需要用户提供信息（如订单号），请友好地询问。"
            ),
            "knowledge": (
                "你正在回答用户的知识性问题。请根据已知信息简洁准确地回答。"
                "如果信息不足，可以告诉用户你暂时无法回答。"
            ),
            "chitchat": (
                "用户正在闲聊。请用友好、自然的语气回复，保持简短。"
            ),
        }

        hint = direction_hints.get(
            result.direction or "", "请根据对话上下文生成合适的回复。"
        )

        # Build a minimal prompt with direction hint
        messages: list[dict] = [
            {
                "role": "system",
                "content": f"你是一个电商客服助手。{hint}回复要简洁、友好、专业。",
            }
        ]

        for msg in state.current_messages:
            role = "user" if msg.role == "user" else "assistant"
            parts: list[str] = []
            if msg.text:
                parts.append(msg.text)
            if msg.object:
                parts.append(
                    f"[{msg.object.type}: {msg.object.title or msg.object.id}]"
                )
            messages.append({"role": role, "content": "\n".join(parts)})

        response = await self._llm.ainvoke(messages)
        text = response.content.strip() if response.content else ""

        return Message(role="bot", text=text)
