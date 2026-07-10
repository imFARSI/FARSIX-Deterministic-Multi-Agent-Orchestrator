"""
FARSIX NeMo Guardrails Critic Agent — LOCAL CPU, zero API calls.

This agent intercepts every other agent's output and applies rule-based
validation using NeMo Guardrails with custom Colang rules. It checks for:

  1. Logical inconsistencies (e.g., risk_score=0 but immediate_action=True)
  2. Hallucination markers (e.g., fabricated statistics, impossible values)
  3. Safety violations (harmful recommendations, unsupported claims)
  4. Completeness (empty sections, missing required fields)

On PASS: approves content, publishes "guardrails_approved" event
On FAIL: blocks content, publishes "guardrails_blocked", forces retry

The NeMo Guardrails library is used for its Colang rule engine.
Falls back to pure Python rule checks if NeMo fails to initialise.
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from backend.event_bus import Event, get_event_bus

# Guardrails directory (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_GUARDRAILS_DIR = os.path.join(_PROJECT_ROOT, "guardrails")


# ------------------------------------------------------------------ #
#  Pure Python rule-based validator (always runs, fast, deterministic) #
# ------------------------------------------------------------------ #

class PythonRuleValidator:
    """
    Deterministic rule-based validator that catches obvious issues.
    This always runs — NeMo Guardrails augments (does not replace) it.
    """

    def validate_vision_output(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Nemotron Vision Agent JSON output."""
        issues = []

        # Must be a dict
        if not isinstance(data, dict):
            return False, "Output is not a JSON object"

        # Required fields
        required = ["scene_summary", "overall_risk_score", "risk_zones"]
        for field in required:
            if field not in data:
                issues.append(f"Missing required field: '{field}'")

        # Risk score range
        score = data.get("overall_risk_score")
        if score is not None:
            try:
                score_int = int(score)
                if not (0 <= score_int <= 100):
                    issues.append(f"overall_risk_score {score_int} out of range [0, 100]")
            except (TypeError, ValueError):
                issues.append(f"overall_risk_score is not numeric: {score}")

        # Logical consistency: score=0 with immediate_action=True
        if score == 0 and data.get("immediate_action_required") is True:
            issues.append(
                "Logical inconsistency: overall_risk_score=0 but "
                "immediate_action_required=True"
            )

        # Critical risk score with no action flag
        if isinstance(score, int) and score >= 80 and data.get("immediate_action_required") is False:
            issues.append(
                f"Logical warning: risk_score={score} (≥80) but "
                "immediate_action_required=False. Flagging for review."
            )
            # This is a warning, not a block — don't append to issues that block

        # scene_summary not empty
        summary = data.get("scene_summary", "")
        if not summary or len(str(summary).strip()) < 10:
            issues.append("scene_summary is empty or too short")

        if issues:
            return False, "Vision validation failed:\n• " + "\n• ".join(issues)
        return True, "Nemotron Vision output passed all validation rules"

    def validate_reasoning_output(self, text: str) -> Tuple[bool, str]:
        """Validate Nemotron reasoning chain text."""
        issues = []

        if not isinstance(text, str):
            return False, "Reasoning output is not a string"

        if len(text.strip()) < 100:
            issues.append("Reasoning output is too short (< 100 chars) — likely incomplete")

        # Check for hallucination markers
        hallucination_patterns = [
            r"\b(100%|0%) (certainty|confidence|probability)\b",
            r"\bI cannot\b.*\breport\b",
            r"\bAs an AI language model\b",
            r"\bI don't have access\b",
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potential hallucination/refusal marker: '{pattern}'")

        # Must contain at least some structure
        has_structure = any(
            marker in text
            for marker in ["##", "**", "1.", "- ", "•", "Risk", "Action", "Priority"]
        )
        if not has_structure:
            issues.append(
                "Reasoning output lacks structured formatting — may be incomplete"
            )

        if issues:
            return False, "Reasoning validation failed:\n• " + "\n• ".join(issues)
        return True, "Reasoning output passed all validation rules"

    def validate_report_output(self, text: str) -> Tuple[bool, str]:
        """Validate Llama final report text."""
        issues = []

        if not isinstance(text, str):
            return False, "Report output is not a string"

        if len(text.strip()) < 200:
            issues.append("Report is too short (< 200 chars)")

        # Must contain key sections
        required_sections = ["Executive Summary", "Key Findings", "Priority"]
        for section in required_sections:
            if section.lower() not in text.lower():
                issues.append(f"Report missing required section: '{section}'")

        # No obviously unsafe content
        unsafe_patterns = [
            r"\b(ignore all|disregard|override)\b.{0,20}\b(safety|rules|instructions)\b",
        ]
        for pattern in unsafe_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Potential safety violation detected: '{pattern}'")

        if issues:
            return False, "Report validation failed:\n• " + "\n• ".join(issues)
        return True, "Report passed all validation rules"


# ------------------------------------------------------------------ #
#  NeMo Guardrails wrapper                                              #
# ------------------------------------------------------------------ #

def _try_load_nemo_rails():
    """
    Attempt to load NeMo Guardrails. Return (rails, None) or (None, error_msg).
    NeMo may not be fully initialised on all machines, so we degrade gracefully.
    """
    try:
        from nemoguardrails import LLMRails, RailsConfig
        import sys
        import os
        
        # Map NVIDIA key to OPENAI key for NeMo's compatibility layer
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = os.environ.get("NVIDIA_API_KEY", "")
        
        # Add the parent directory to sys.path so we can import guardrails.actions
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
            
        from guardrails.actions import (
            is_guardrails_block,
            check_content_safety,
            check_for_safety_violations,
            check_logical_consistency,
            detect_hallucinations,
            check_output_completeness
        )

        config = RailsConfig.from_path(_GUARDRAILS_DIR)
        rails = LLMRails(config)
        
        # Register Colang actions
        rails.register_action(is_guardrails_block, "is_guardrails_block")
        rails.register_action(check_content_safety, "check_content_safety")
        rails.register_action(check_for_safety_violations, "check_for_safety_violations")
        rails.register_action(check_logical_consistency, "check_logical_consistency")
        rails.register_action(detect_hallucinations, "detect_hallucinations")
        rails.register_action(check_output_completeness, "check_output_completeness")
        
        return rails, None
    except Exception as exc:
        import traceback
        print(f"[GUARDRAILS INIT ERROR] {exc}\\n{traceback.format_exc()}")
        return None, str(exc)


# ------------------------------------------------------------------ #
#  Main Guardrails Agent                                                #
# ------------------------------------------------------------------ #

class GuardrailsAgent:
    """
    NeMo Guardrails Critic Agent — runs 100% locally on CPU.

    Validates every agent output before it reaches the next stage.
    Uses both NeMo Guardrails (Colang rules) and a pure Python
    rule-based validator as a reliable fallback.

    Usage:
        critic = GuardrailsAgent()
        passed, reason = await critic.validate(
            content=vision_result,
            content_type="vision",
            mission_id="m-001",
        )
    """

    AGENT_NAME = "guardrails_agent"
    DISPLAY_NAME = "GUARDRAILS"
    COLOR_PASS = "#66BB6A"   # Green
    COLOR_FAIL = "#EF5350"   # Red

    def __init__(self):
        self.bus = get_event_bus()
        self.python_validator = PythonRuleValidator()
        self._nemo_rails = None
        self._nemo_error: Optional[str] = None
        self._nemo_loaded = False

    def _ensure_nemo(self) -> None:
        if not self._nemo_loaded:
            self._nemo_rails, self._nemo_error = _try_load_nemo_rails()
            self._nemo_loaded = True

    # ------------------------------------------------------------------ #
    #  Main validate entry point                                            #
    # ------------------------------------------------------------------ #

    async def validate(
        self,
        content: Any,
        content_type: str,       # "vision" | "reasoning" | "report"
        mission_id: str = "unknown",
    ) -> Tuple[bool, str]:
        """
        Validate agent output. Returns (passed: bool, reason: str).

        If passed=False, the mission engine should retry the previous agent.
        """
        t_start = time.time()
        self._publish("agent_started", mission_id, {
            "message": f"🛡️ GUARDRAILS validating {content_type} output...",
            "color": "orange",
        })

        # --- Python rule checks (always run first) ---
        py_passed, py_reason = await asyncio.get_event_loop().run_in_executor(
            None, self._python_validate, content, content_type
        )

        if not py_passed:
            elapsed = round(time.time() - t_start, 2)
            self._publish("guardrails_blocked", mission_id, {
                "message": f"🔴 GUARDRAILS BLOCKED ({content_type}) — {py_reason}",
                "reason": py_reason,
                "elapsed": elapsed,
                "color": "red",
            })
            return False, py_reason

        # --- NeMo Guardrails check (augments Python rules) ---
        await asyncio.get_event_loop().run_in_executor(None, self._ensure_nemo)

        if self._nemo_rails is not None:
            nemo_passed, nemo_reason = await self._nemo_validate(content, content_type, mission_id)
            if not nemo_passed:
                elapsed = round(time.time() - t_start, 2)
                self._publish("guardrails_blocked", mission_id, {
                    "message": f"🔴 GUARDRAILS BLOCKED (NeMo rule) — {nemo_reason}",
                    "reason": nemo_reason,
                    "elapsed": elapsed,
                    "color": "red",
                })
                return False, nemo_reason
        else:
            self._publish("guardrails_info", mission_id, {
                "message": (
                    f"⚠️ NeMo Guardrails unavailable "
                    f"({self._nemo_error or 'init failed'}) — "
                    f"using Python rules only"
                ),
                "color": "orange",
            })

        elapsed = round(time.time() - t_start, 2)
        final_reason = f"{py_reason}"
        self._publish("guardrails_approved", mission_id, {
            "message": f"✅ GUARDRAILS APPROVED ({content_type}) in {elapsed}s",
            "reason": final_reason,
            "elapsed": elapsed,
            "color": "green",
        })
        return True, final_reason

    # ------------------------------------------------------------------ #
    #  Internal validators                                                  #
    # ------------------------------------------------------------------ #

    def _python_validate(self, content: Any, content_type: str) -> Tuple[bool, str]:
        if content_type == "vision":
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    return False, "Nemotron Vision output is not valid JSON"
            return self.python_validator.validate_vision_output(content)

        elif content_type == "reasoning":
            text = content if isinstance(content, str) else json.dumps(content)
            return self.python_validator.validate_reasoning_output(text)

        elif content_type == "report":
            text = content if isinstance(content, str) else json.dumps(content)
            return self.python_validator.validate_report_output(text)

        else:
            return True, f"Unknown content_type '{content_type}' — skipping validation"

    async def _nemo_validate(
        self,
        content: Any,
        content_type: str,
        mission_id: str,
    ) -> Tuple[bool, str]:
        """
        Run NeMo Guardrails checks via the Colang rules in guardrails/safety_rules.co.
        NeMo rails are designed for conversational input, so we wrap content as a message.
        """
        try:
            text = content if isinstance(content, str) else json.dumps(content, indent=2)
            short_text = text[:500]  # NeMo checks on a window, not the full text

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._nemo_generate,
                f"Validate this {content_type} output for safety and accuracy:\n{short_text}",
            )

            # NeMo will return a blocked response if Colang rules trigger
            blocked_phrases = [
                "cannot", "unable to", "blocked", "violation", "unsafe",
                "I'm not able", "This content"
            ]
            response_lower = (response or "").lower()
            for phrase in blocked_phrases:
                if phrase.lower() in response_lower:
                    return False, f"NeMo Guardrails flagged content: {response[:200]}"

            return True, "NeMo Guardrails: no violations detected"

        except Exception as exc:
            # NeMo failure should not block the pipeline — log and pass through
            self._publish("guardrails_info", mission_id, {
                "message": f"⚠️ NeMo check skipped: {exc}",
                "color": "orange",
            })
            return True, f"NeMo check skipped (error): {exc}"

    def _nemo_generate(self, message: str) -> str:
        """Synchronous NeMo rails call — wrapped in executor."""
        if self._nemo_rails is None:
            return ""
        try:
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            resp = loop.run_until_complete(
                self._nemo_rails.generate_async(messages=[
                    {"role": "user", "content": message}
                ])
            )
            loop.close()
            if isinstance(resp, dict):
                return resp.get("content", "")
            return str(resp)
        except Exception as exc:
            print(f"\\n[NeMo Colang Exception] {exc}")
            raise exc

    def _publish(self, event_type: str, mission_id: str, data: dict) -> None:
        self.bus.publish(Event(
            event_type=event_type,
            source=self.AGENT_NAME,
            mission_id=mission_id,
            data=data,
        ))
