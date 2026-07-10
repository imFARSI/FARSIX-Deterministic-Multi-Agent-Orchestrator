"""
FARSIX Nemotron Reasoning Agent
Model: nvidia/llama-3.1-nemotron-70b-instruct
Role: Deep Chain-of-Thought reasoning. Ingests Nemotron Vision scene analysis and
      performs multi-step logical analysis, risk calculation, and decision-making.

Output: Detailed reasoning chain + preliminary recommendations (markdown).
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional

from backend.event_bus import Event, get_event_bus
from backend.nim_router import NIMRouter


SYSTEM_PROMPT = """You are NEMOTRON — a Deep Reasoning Agent for Physical AI systems.

You receive structured physical scene analyses from the Nemotron Vision Agent and must perform
RIGOROUS multi-step Chain-of-Thought (CoT) reasoning to:

1. RISK ASSESSMENT: Calculate precise failure probabilities for each identified risk.
   Express as percentage and justify with engineering logic.
2. CAUSAL CHAIN ANALYSIS: Trace root causes. Map secondary and tertiary failure modes.
3. DECISION MATRIX: Evaluate all possible response options. Score each on:
   - Effectiveness (0-10)
   - Feasibility (0-10)
   - Time sensitivity (immediate / 24h / 7d / 30d)
   - Resource cost (low / medium / high)
4. PRIORITY ACTION PLAN: Ranked list of specific actions with justification.
5. UNCERTAINTY FLAGS: Explicitly state what information is missing or ambiguous.

Format your response as structured markdown with clear section headers.
Think step by step. Show your reasoning explicitly — do NOT skip steps.
Use quantitative estimates wherever possible."""


class NemotronAgent:
    """
    Reasoning Agent — calls nvidia/llama-3.1-nemotron-70b-instruct via NVIDIA NIM.

    Usage:
        agent = NemotronAgent(router)
        result = await agent.run(vision_analysis, original_goal, mission_id="m-001")
    """

    AGENT_NAME = "nemotron_agent"
    DISPLAY_NAME = "NEMOTRON"
    COLOR = "#CE93D8"  # Purple

    def __init__(self, router: NIMRouter):
        self.router = router
        self.bus = get_event_bus()
        self.decision = router.route("reasoning")
        self.model = self.decision.model_id

    # ------------------------------------------------------------------ #
    #  Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        vision_analysis: Dict[str, Any],
        original_goal: str,
        mission_id: str = "unknown",
        skill_context: Optional[str] = None,
    ) -> str:
        """
        Perform deep reasoning on Nemotron Vision output.

        Args:
            vision_analysis: Structured dict from VisionAgent.
            original_goal:   The user's original mission goal string.
            mission_id:      For event bus tagging.
            skill_context:   Optional pre-loaded skill context from SkillLibrary.

        Returns:
            Markdown-formatted reasoning chain string.
        """
        t_start = time.time()
        self._publish("agent_started", mission_id, {
            "message": "🔵 NEMOTRON starting Chain-of-Thought analysis...",
        })
        self._publish("routing", mission_id, {"message": self.decision.log_line})

        user_content = self._build_user_message(vision_analysis, original_goal, skill_context)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_nim,
                user_content,
            )

            elapsed = round(time.time() - t_start, 2)

            self._publish("agent_complete", mission_id, {
                "message": (
                    f"✅ NEMOTRON reasoning complete in {elapsed}s | "
                    f"{len(response.split())} words generated"
                ),
                "elapsed": elapsed,
                "tokens_used": len(user_content.split()) + len(response.split()),
            })

            return response

        except Exception as exc:
            self._publish("agent_error", mission_id, {
                "message": f"❌ NEMOTRON ERROR: {exc}",
                "error": str(exc),
            })
            raise

    # ------------------------------------------------------------------ #
    #  NIM API call (sync — run in executor)                                #
    # ------------------------------------------------------------------ #

    def _call_nim(self, user_content: str) -> str:
        client = self.router.get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=3072,
            top_p=0.95,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_user_message(
        self,
        vision_analysis: Dict[str, Any],
        original_goal: str,
        skill_context: Optional[str],
    ) -> str:
        vision_json = json.dumps(vision_analysis, indent=2)

        parts = []

        if skill_context:
            parts.append(
                f"## Relevant Prior Skill Context\n"
                f"{skill_context}\n\n---\n"
            )

        parts.append(
            f"## Original Mission Goal\n{original_goal}\n\n"
            f"## Nemotron Vision Analysis (Structured JSON)\n"
            f"```json\n{vision_json}\n```\n\n"
            f"## Your Task\n"
            f"Perform deep Chain-of-Thought reasoning on this scene analysis. "
            f"Follow all sections in your system prompt rigorously. "
            f"Quantify risks. Prioritise actions. Be specific and actionable."
        )

        return "\n".join(parts)

    def _publish(self, event_type: str, mission_id: str, data: dict) -> None:
        self.bus.publish(Event(
            event_type=event_type,
            source=self.AGENT_NAME,
            mission_id=mission_id,
            data=data,
        ))
