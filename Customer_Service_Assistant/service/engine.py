"""Engine layer — core dispatcher that orchestrates a single turn.

Flow:  Prepare Session → Create Turn → Classify Message →
       (Object | Text) → Planning → Generate → Commit
"""

from __future__ import annotations

import time

from Customer_Service_Assistant.infrastructure.llm import llm
from Customer_Service_Assistant.service.schemas import DialogueState, Message, Turn

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

    # -- public API ----------------------------------------------------------

    async def run(self, state: DialogueState, user_message: Message) -> Message:
        """Process *state* with the incoming *user_message* and return the
        bot's reply as a ``Message``.

        Steps (per DialogueEngine 设计.md):

        1. Prepare Session
        2. Create Turn
        3. Classify message — text, object, or both
        4. Process object — write focused_object, check slot match
        5. Process text — planning → generate
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
            # Step 5 — text messages: plan → generate via LLM
            prompt = self._plan(state)
            bot_text = await self._generate(prompt)
            bot_msg = Message(role="bot", text=bot_text)
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

    # -- Step 5: process text messages (Planning) ---------------------------

    def _plan(self, state: DialogueState) -> list[dict]:
        """Build the LLM prompt from the full conversation history **and**
        the current running state (active_task, paused_tasks, focused_object).

        Design doc §2.5 — TurnPlanner should consider:

        * Recent dialogue
        * ``active_task`` — is the user continuing a previous task?
        * ``paused_tasks`` — is the user resuming a paused task?
        * ``focused_object`` — order / product context the user clicked
        * Available flows & knowledge intents (stub — hard-coded below)
        """
        system_parts: list[str] = [
            "你是一个电商客服助手。请根据用户的问题提供帮助。",
            "回复要简洁、友好、专业。",
        ]

        # -- Context: active task --------------------------------------------
        active = state.active_task
        if active is not None:
            system_parts.append(
                f"当前正在处理：{active.flow_id}（步骤 {active.step_id}）。"
                f"用户可能在继续此任务。"
            )

        # -- Context: paused tasks -------------------------------------------
        paused = state.paused_tasks
        if paused:
            names = ", ".join(p.flow_id for p in paused)
            system_parts.append(
                f"用户有暂停的任务：{names}。用户可能想回到其中之一。"
            )

        # -- Context: focused object -----------------------------------------
        focus = state.focused_object
        if focus is not None:
            label = focus.title or focus.id
            system_parts.append(
                f"用户当前关注的对象：[{focus.type}] {label}。"
            )

        # -- Context: available capabilities (stub) --------------------------
        system_parts.append(
            "你可以处理以下业务：查订单、查物流、申请退款、催发货、推荐商品。"
            "你可以回答以下知识类问题：商品信息、退换货政策、常见问题。"
        )

        prompt: list[dict] = [
            {"role": "system", "content": "\n".join(system_parts)}
        ]

        # -- Recent dialogue -------------------------------------------------
        for msg in state.current_messages:
            role = "user" if msg.role == "user" else "assistant"
            parts: list[str] = []
            if msg.text:
                parts.append(msg.text)
            if msg.object:
                parts.append(
                    f"[{msg.object.type}: {msg.object.title or msg.object.id}]"
                )
            prompt.append({"role": role, "content": "\n".join(parts)})

        return prompt

    # -- Generate (stub — will fan out to tracks) ----------------------------

    async def _generate(self, prompt: list[dict]) -> str:
        """Call the LLM and return the response text.

        When Route is built, this will become a dispatch to
        Task / Knowledge / Chitchat tracks.
        """
        response = await self._llm.ainvoke(prompt)
        return response.content.strip() if response.content else ""
