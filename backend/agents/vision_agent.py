"""
FARSIX Nemotron Vision Agent
Model: nvidia/llama-3.1-nemotron-nano-vl-8b-v1
Role: Physical world understanding. Extracts physical properties, spatial
      relationships, object states, risk zones, and cause-effect relationships.

Output: Structured JSON scene analysis (parsed into dict).
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, Optional

from backend.event_bus import Event, get_event_bus
from backend.nim_router import NIMRouter


SYSTEM_PROMPT = """You are NEMOTRON VISION — a Physical AI Vision Agent specialising in
understanding the physical world. Your role is to analyse scene descriptions,
sensor data, or image contexts and extract:

1. PHYSICAL PROPERTIES: measurable quantities, sensor readings, material states
2. SPATIAL RELATIONSHIPS: positional context, proximity, layout
3. OBJECT STATES: condition (normal / degraded / critical / failed), change over time
4. RISK ZONES: areas or components at elevated risk, severity levels (LOW / MEDIUM / HIGH / CRITICAL)
5. CAUSE-EFFECT RELATIONSHIPS: causal chains that explain observed anomalies
6. ANOMALIES: deviations from expected baselines with quantified deltas

Return ONLY a valid JSON object in this exact schema — no markdown fences, no extra text:
{
  "scene_summary": "<one-sentence overall description>",
  "physical_properties": [{"name": str, "value": str, "unit": str, "status": str}],
  "spatial_relationships": [{"subject": str, "relation": str, "object": str}],
  "object_states": [{"object": str, "state": str, "severity": str, "detail": str}],
  "risk_zones": [{"zone": str, "risk_level": str, "description": str, "probability": str}],
  "cause_effect": [{"cause": str, "effect": str, "confidence": str}],
  "anomalies": [{"component": str, "expected": str, "observed": str, "delta": str, "severity": str}],
  "overall_risk_score": <integer 0-100>,
  "immediate_action_required": <boolean>,
  "raw_observations": "<detailed paragraph of all observations>"
}"""


class VisionAgent:
    """
    Nemotron Vision Agent — calls nvidia/llama-3.1-nemotron-nano-vl-8b-v1 via NVIDIA NIM.

    Usage:
        agent = VisionAgent(router)
        result = await agent.run(input_text, mission_id="m-001")
    """

    AGENT_NAME = "vision_agent"
    DISPLAY_NAME = "NEMOTRON VISION"
    COLOR = "#4FC3F7"  # Blue

    def __init__(self, router: NIMRouter):
        self.router = router
        self.bus = get_event_bus()
        self.decision = router.route("vision")
        self.model = self.decision.model_id

    # ------------------------------------------------------------------ #
    #  Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        input_text: str,
        input_image_b64: Optional[str] = None,
        mission_id: str = "unknown",
        memory_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyse input_text and return a structured physical scene dict.

        Args:
            input_text:     Raw scene description, extracted PDF/CSV text, or image caption.
            mission_id:     For event bus tagging.
            memory_context: Optional injected context from ChromaDB memory.

        Returns:
            Parsed JSON dict from the model.
        """
        t_start = time.time()
        self._publish("agent_started", mission_id, {"message": f"🔵 VISION starting scene analysis..."})

        # Build user message
        user_content = self._build_user_message(input_text, memory_context)

        # Route log
        self._publish("routing", mission_id, {"message": self.decision.log_line})

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_nim,
                user_content,
                input_image_b64,
            )

            elapsed = round(time.time() - t_start, 2)
            result = self._parse_response(response)

            self._publish("agent_complete", mission_id, {
                "message": (
                    f"✅ VISION completed in {elapsed}s | "
                    f"Risk score: {result.get('overall_risk_score', '?')}/100 | "
                    f"Anomalies: {len(result.get('anomalies', []))}"
                ),
                "result_summary": result.get("scene_summary", ""),
                "elapsed": elapsed,
                "tokens_used": len(user_content.split()) + len(response.split()),
            })

            return result

        except Exception as exc:
            self._publish("agent_error", mission_id, {
                "message": f"❌ VISION ERROR: {exc}",
                "error": str(exc),
            })
            raise

    # ------------------------------------------------------------------ #
    #  NIM API call (sync — run in executor)                                #
    # ------------------------------------------------------------------ #

    def _call_nim(self, user_content: str, input_image_b64: Optional[str] = None) -> str:
        client = self.router.get_client()
        
        msg_content = [{"type": "text", "text": user_content}]
        if input_image_b64:
            msg_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{input_image_b64}"}
            })

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg_content},
            ],
            temperature=0.2,
            max_tokens=2048,
            top_p=0.9,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_user_message(self, input_text: str,
                             memory_context: Optional[str]) -> str:
        parts = []
        if memory_context:
            parts.append(
                f"[MEMORY CONTEXT — similar past mission]\n{memory_context}\n"
                f"---\n[END MEMORY CONTEXT]\n"
            )
        parts.append(
            f"Analyse the following scene and return the structured JSON output:\n\n"
            f"{input_text}"
        )
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Robustly parse the model's JSON response.
        Falls back to a minimal dict if JSON is malformed.
        """
        # Strip any accidental markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Fallback: wrap raw text in minimal structure
        return {
            "scene_summary": "Analysis completed (JSON parse error — raw text preserved)",
            "physical_properties": [],
            "spatial_relationships": [],
            "object_states": [],
            "risk_zones": [],
            "cause_effect": [],
            "anomalies": [],
            "overall_risk_score": 50,
            "immediate_action_required": False,
            "raw_observations": raw,
        }

    def _publish(self, event_type: str, mission_id: str, data: dict) -> None:
        self.bus.publish(Event(
            event_type=event_type,
            source=self.AGENT_NAME,
            mission_id=mission_id,
            data=data,
        ))
