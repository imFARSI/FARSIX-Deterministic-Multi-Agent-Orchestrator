"""
FARSIX Mission Engine — State machine + mission queue manager.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.agents.vision_agent import VisionAgent
from backend.agents.guardrails_agent import GuardrailsAgent
from backend.agents.llama_agent import LlamaAgent
from backend.agents.nemotron_agent import NemotronAgent
from backend.event_bus import Event, get_event_bus
from backend.nim_router import NIMRouter
from backend.recovery import RecoveryEngine, get_recovery_engine
from backend.skill_library import NewSkillPayload, SkillLibrary, get_skill_library


# ------------------------------------------------------------------ #
#  State machine                                                        #
# ------------------------------------------------------------------ #

class MissionState(str, Enum):
    QUEUED          = "QUEUED"
    PARSING_INPUT   = "PARSING_INPUT"
    VISION_ANALYSIS = "VISION_ANALYSIS"
    DEEP_REASONING  = "DEEP_REASONING"
    VALIDATION      = "VALIDATION"
    SUMMARIZING     = "SUMMARIZING"
    COMPLETE        = "COMPLETE"
    FAILED          = "FAILED"


STATE_ORDER = [
    MissionState.QUEUED,
    MissionState.PARSING_INPUT,
    MissionState.VISION_ANALYSIS,
    MissionState.DEEP_REASONING,
    MissionState.VALIDATION,
    MissionState.SUMMARIZING,
    MissionState.COMPLETE,
]


# ------------------------------------------------------------------ #
#  Mission dataclass                                                    #
# ------------------------------------------------------------------ #

@dataclass
class Mission:
    id: str
    goal: str
    input_type: str          # "text" | "pdf" | "csv" | "image"
    input_text: str          # Parsed/extracted text content
    input_image_b64: Optional[str] = None # Base64 encoded image
    state: MissionState = MissionState.QUEUED
    retry_count: int = 0
    max_retries: int = 2
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Agent outputs
    vision_result: Optional[Dict[str, Any]] = None
    nemotron_result: Optional[str] = None
    guardrails_passed: Optional[bool] = None
    final_report: Optional[str] = None

    # Metadata
    memory_context: Optional[str] = None
    skill_context: Optional[str] = None
    matched_skill_id: Optional[int] = None
    error_message: Optional[str] = None
    token_count: int = 0

    def elapsed(self) -> float:
        if self.started_at:
            end = self.completed_at or time.time()
            return round(end - self.started_at, 1)
        return 0.0

    def progress_pct(self) -> int:
        """Return 0-100 progress percentage based on current state."""
        progress_map = {
            MissionState.QUEUED:          0,
            MissionState.PARSING_INPUT:   10,
            MissionState.VISION_ANALYSIS: 30,
            MissionState.DEEP_REASONING:  55,
            MissionState.VALIDATION:      70,
            MissionState.SUMMARIZING:     85,
            MissionState.COMPLETE:        100,
            MissionState.FAILED:          100,
        }
        return progress_map.get(self.state, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "input_type": self.input_type,
            "state": self.state.value,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": self.elapsed(),
            "progress_pct": self.progress_pct(),
            "final_report": self.final_report,
            "error_message": self.error_message,
            "vision_result": self.vision_result,
            "token_count": self.token_count,
            "memory_context": self.memory_context,
            "matched_skill_id": self.matched_skill_id,
        }


# ------------------------------------------------------------------ #
#  Mission Engine                                                       #
# ------------------------------------------------------------------ #

class MissionEngine:
    """
    Orchestrates multi-agent missions through the state machine.

    Usage:
        engine = MissionEngine()
        mission = engine.create_mission(goal, input_type, input_text)
        asyncio.run(engine.run_mission(mission.id))
        result = engine.get_mission(mission.id)
    """

    def __init__(self):
        self.bus = get_event_bus()
        self.router = NIMRouter()
        self.recovery = get_recovery_engine()
        self.skill_lib = get_skill_library()

        # Lazy-init agents (created on first use)
        self._vision: Optional[VisionAgent] = None
        self._nemotron: Optional[NemotronAgent] = None
        self._llama: Optional[LlamaAgent] = None
        self._guardrails: Optional[GuardrailsAgent] = None

        # In-memory mission registry
        self._missions: Dict[str, Mission] = {}

        # ChromaDB store (lazy import)
        self._chroma = None

    # ------------------------------------------------------------------ #
    #  Agent accessors (lazy init)                                          #
    # ------------------------------------------------------------------ #

    def _get_vision(self) -> VisionAgent:
        if self._vision is None:
            self._vision = VisionAgent(self.router)
        return self._vision

    def _get_nemotron(self) -> NemotronAgent:
        if self._nemotron is None:
            self._nemotron = NemotronAgent(self.router)
        return self._nemotron

    def _get_llama(self) -> LlamaAgent:
        if self._llama is None:
            self._llama = LlamaAgent(self.router)
        return self._llama

    def _get_guardrails(self) -> GuardrailsAgent:
        if self._guardrails is None:
            self._guardrails = GuardrailsAgent()
        return self._guardrails

    def _get_chroma(self):
        if self._chroma is None:
            try:
                from memory.chroma_store import get_chroma_store
                self._chroma = get_chroma_store()
            except Exception:
                self._chroma = None
        return self._chroma

    # ------------------------------------------------------------------ #
    #  Mission CRUD                                                         #
    # ------------------------------------------------------------------ #

    def create_mission(
        self,
        goal: str,
        input_type: str,
        input_text: str,
        input_image_b64: Optional[str] = None,
    ) -> Mission:
        mission_id = f"m-{uuid.uuid4().hex[:8]}"
        mission = Mission(
            id=mission_id,
            goal=goal,
            input_type=input_type,
            input_text=input_text,
            input_image_b64=input_image_b64,
        )
        self._missions[mission_id] = mission
        self.skill_lib.save_mission(mission_id, goal, input_type, "QUEUED")
        self._publish_state(mission)
        return mission

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        return self._missions.get(mission_id)

    def list_missions(self) -> List[Mission]:
        return sorted(self._missions.values(), key=lambda m: m.created_at, reverse=True)

    def get_active_agent_count(self) -> int:
        statuses = self.recovery.all_statuses()
        return sum(1 for s in statuses.values() if s == "ONLINE")

    # ------------------------------------------------------------------ #
    #  Main mission runner                                                  #
    # ------------------------------------------------------------------ #

    async def run_mission(self, mission_id: str) -> Mission:
        """
        Execute a full mission through the state machine.
        Returns the completed (or failed) Mission object.
        """
        mission = self._missions.get(mission_id)
        if not mission:
            raise ValueError(f"Mission {mission_id} not found")

        mission.started_at = time.time()

        for attempt in range(mission.max_retries + 1):
            try:
                mission = await self._execute_pipeline(mission)
                break  # Success
            except Exception as exc:
                mission.retry_count = attempt + 1
                mission.error_message = str(exc)
                if attempt < mission.max_retries:
                    self._publish_event("mission_retry", mission.id, {
                        "attempt": attempt + 1,
                        "max": mission.max_retries,
                        "message": (
                            f"⟳ MISSION RETRY {attempt + 1}/{mission.max_retries} — "
                            f"restarting from last checkpoint..."
                        ),
                    })
                    # Reset to last successful checkpoint
                    if mission.guardrails_passed:
                        mission.state = MissionState.SUMMARIZING
                    elif mission.vision_result is not None:
                        mission.state = MissionState.DEEP_REASONING
                    else:
                        mission.state = MissionState.QUEUED
                    await asyncio.sleep(2)
                else:
                    mission.state = MissionState.FAILED
                    mission.completed_at = time.time()
                    self._publish_state(mission)
                    self.skill_lib.save_mission(
                        mission.id, mission.goal, mission.input_type,
                        "FAILED", result=mission.error_message,
                        retry_count=mission.retry_count,
                    )

        return mission

    # ------------------------------------------------------------------ #
    #  Pipeline stages                                                      #
    # ------------------------------------------------------------------ #

    async def _execute_pipeline(self, mission: Mission) -> Mission:
        """Run through state machine from current state to COMPLETE."""

        # --- STAGE: PARSING_INPUT ---
        if mission.state in (MissionState.QUEUED, MissionState.PARSING_INPUT):
            await self._stage_parse(mission)

        # --- STAGE: VISION_ANALYSIS ---
        if mission.state == MissionState.VISION_ANALYSIS:
            await self._stage_vision(mission)

        # --- STAGE: DEEP_REASONING ---
        if mission.state == MissionState.DEEP_REASONING:
            await self._stage_reasoning(mission)

        # --- STAGE: VALIDATION ---
        if mission.state == MissionState.VALIDATION:
            await self._stage_validation(mission)

        # --- STAGE: SUMMARIZING ---
        if mission.state == MissionState.SUMMARIZING:
            await self._stage_summarize(mission)

        # --- STAGE: COMPLETE ---
        if mission.state == MissionState.COMPLETE:
            await self._stage_complete(mission)

        return mission

    async def _stage_parse(self, mission: Mission) -> None:
        self._transition(mission, MissionState.PARSING_INPUT, "Parsing input content")

        # ChromaDB memory lookup
        chroma = self._get_chroma()
        if chroma:
            try:
                similar = chroma.search(mission.input_text, n_results=1)
                if similar and similar[0]["similarity"] >= 0.85:
                    mission.memory_context = similar[0]["content"]
                    self._publish_event("memory_hit", mission.id, {
                        "message": (
                            f"🧠 Memory Agent: Found similar mission "
                            f"({round(similar[0]['similarity'] * 100)}% match). "
                            f"Context injected."
                        ),
                        "similarity": similar[0]["similarity"],
                    })
            except Exception:
                pass  # Memory failure is non-fatal

        # Skill library lookup
        skills = self.skill_lib.search_skills(mission.goal, input_type=mission.input_type)
        if skills:
            top_skill = skills[0]
            mission.matched_skill_id = top_skill.id
            mission.skill_context = (
                f"[PRIOR SKILL: {top_skill.name}]\n"
                f"{top_skill.context_snapshot}"
            )
            self._publish_event("skill_matched", mission.id, {
                "message": (
                    f"📚 Skill Library: Matched '{top_skill.name}' "
                    f"(used {top_skill.usage_count}×, "
                    f"{round(top_skill.success_rate * 100)}% success). "
                    f"Context pre-loaded."
                ),
                "skill_name": top_skill.name,
            })

        self._transition(mission, MissionState.VISION_ANALYSIS, "Input parsed, starting vision analysis")

    async def _stage_vision(self, mission: Mission) -> None:
        self._transition(mission, MissionState.VISION_ANALYSIS, "Running Nemotron Vision analysis")
        vision = self._get_vision()

        vision_result = await self.recovery.run_with_recovery(
            coro_factory=lambda: vision.run(
                input_text=mission.input_text,
                input_image_b64=mission.input_image_b64,
                mission_id=mission.id,
                memory_context=mission.memory_context,
            ),
            agent_name="vision_agent",
            mission_id=mission.id,
            timeout=120,
        )

        mission.vision_result = vision_result
        self._transition(mission, MissionState.DEEP_REASONING, "Vision complete, starting reasoning")

    async def _stage_reasoning(self, mission: Mission) -> None:
        self._transition(mission, MissionState.DEEP_REASONING, "Running Nemotron CoT reasoning")
        nemotron = self._get_nemotron()

        reasoning = await self.recovery.run_with_recovery(
            coro_factory=lambda: nemotron.run(
                vision_analysis=mission.vision_result or {},
                original_goal=mission.goal,
                mission_id=mission.id,
                skill_context=mission.skill_context,
            ),
            agent_name="nemotron_agent",
            mission_id=mission.id,
            timeout=120,
        )

        mission.nemotron_result = reasoning
        mission.token_count += len(reasoning.split())
        self._transition(mission, MissionState.VALIDATION, "Reasoning complete, starting validation")

    async def _stage_validation(self, mission: Mission) -> None:
        self._transition(mission, MissionState.VALIDATION, "Running NeMo Guardrails validation")
        guardrails = self._get_guardrails()

        # Validate vision output
        vision_ok, vision_reason = await guardrails.validate(
            content=mission.vision_result or {},
            content_type="vision",
            mission_id=mission.id,
        )
        if not vision_ok:
            raise ValueError(f"Vision validation failed: {vision_reason}")

        # Validate reasoning output
        reasoning_ok, reasoning_reason = await guardrails.validate(
            content=mission.nemotron_result or "",
            content_type="reasoning",
            mission_id=mission.id,
        )
        if not reasoning_ok:
            raise ValueError(f"Reasoning validation failed: {reasoning_reason}")

        mission.guardrails_passed = True
        self._transition(mission, MissionState.SUMMARIZING, "Validation passed, generating report")

    async def _stage_summarize(self, mission: Mission) -> None:
        self._transition(mission, MissionState.SUMMARIZING, "Running Llama report generation")
        llama = self._get_llama()

        report = await self.recovery.run_with_recovery(
            coro_factory=lambda: llama.run(
                reasoning_chain=mission.nemotron_result or "",
                vision_summary=mission.vision_result,
                mission_id=mission.id,
            ),
            agent_name="llama_agent",
            mission_id=mission.id,
            timeout=120,
        )

        # Validate report
        guardrails = self._get_guardrails()
        report_ok, report_reason = await guardrails.validate(
            content=report,
            content_type="report",
            mission_id=mission.id,
        )
        if not report_ok:
            raise ValueError(f"Report validation failed: {report_reason}")

        mission.final_report = report
        mission.token_count += len(report.split())
        self._transition(mission, MissionState.COMPLETE, "Report complete")

    async def _stage_complete(self, mission: Mission) -> None:
        mission.completed_at = time.time()

        # Save to ChromaDB
        chroma = self._get_chroma()
        if chroma and mission.final_report:
            try:
                chroma.add(
                    content=mission.input_text + "\n\n" + (mission.final_report or ""),
                    metadata={
                        "mission_id": mission.id,
                        "input_type": mission.input_type,
                        "goal": mission.goal,
                    },
                )
            except Exception:
                pass

        # Save skill
        skill_name = self._derive_skill_name(mission)
        context_snap = json.dumps({
            "goal": mission.goal,
            "vision_summary": (mission.vision_result or {}).get("scene_summary", ""),
            "risk_score": (mission.vision_result or {}).get("overall_risk_score", 0),
            "report_preview": (mission.final_report or "")[:300],
        })
        new_skill = self.skill_lib.save_skill(NewSkillPayload(
            name=skill_name,
            input_type=mission.input_type,
            description=mission.goal[:200],
            context_snapshot=context_snap,
        ))

        if mission.matched_skill_id:
            self.skill_lib.increment_usage(mission.matched_skill_id, success=True)

        # Save mission to DB
        self.skill_lib.save_mission(
            mission.id, mission.goal, mission.input_type,
            "COMPLETE", result=mission.final_report,
            retry_count=mission.retry_count,
        )

        self._publish_event("mission_complete", mission.id, {
            "message": (
                f"🏆 MISSION COMPLETE in {mission.elapsed()}s | "
                f"Tokens: {mission.token_count}"
            ),
            "elapsed": mission.elapsed(),
            "token_count": mission.token_count,
            "skill_name": new_skill.name,
            "new_skill_notification": f"🆕 NEW SKILL LEARNED: {new_skill.name}",
        })

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _transition(self, mission: Mission, new_state: MissionState, detail: str) -> None:
        old_state = mission.state
        mission.state = new_state
        mission.updated_at = time.time()
        self._publish_event("mission_state_change", mission.id, {
            "old_state": old_state.value,
            "new_state": new_state.value,
            "detail": detail,
            "message": f"📍 {old_state.value} → {new_state.value}: {detail}",
        })
        self.skill_lib.save_mission(
            mission.id, mission.goal, mission.input_type,
            new_state.value, retry_count=mission.retry_count,
        )

    def _publish_state(self, mission: Mission) -> None:
        self._publish_event("mission_state_change", mission.id, {
            "new_state": mission.state.value,
            "message": f"📍 Mission state: {mission.state.value}",
        })

    def _publish_event(self, event_type: str, mission_id: str, data: dict) -> None:
        self.bus.publish(Event(
            event_type=event_type,
            source="mission_engine",
            mission_id=mission_id,
            data=data,
        ))

    def _derive_skill_name(self, mission: Mission) -> str:
        """Generate a concise skill name from the mission goal."""
        goal = mission.goal.strip()
        # Take first 6 meaningful words
        words = [w for w in goal.split() if len(w) > 3][:6]
        name = " ".join(words).title()
        return name if name else f"Mission {mission.id[:8]}"


# ------------------------------------------------------------------ #
#  Singleton                                                            #
# ------------------------------------------------------------------ #

_engine: Optional[MissionEngine] = None


def get_mission_engine() -> MissionEngine:
    global _engine
    if _engine is None:
        _engine = MissionEngine()
    return _engine
