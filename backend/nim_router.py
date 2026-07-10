"""
FARSIX NIM Router — Intelligent model selector for NVIDIA NIM API calls.

The router inspects task_type and automatically maps it to the correct NIM model,
then returns a configured OpenAI client pointed at the NVIDIA base URL.

Visual indicator strings are returned for the dashboard to display routing decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ------------------------------------------------------------------ #
#  Model registry                                                       #
# ------------------------------------------------------------------ #

MODELS = {
    "vision": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "reasoning": "meta/llama-3.1-70b-instruct",
    "summarize": "meta/llama-3.1-8b-instruct",
}

MODEL_DESCRIPTIONS = {
    "vision": "Physical scene understanding & spatial analysis",
    "reasoning": "Deep Chain-of-Thought analysis & risk calculation",
    "summarize": "Fast summarization & executive report generation",
}

AGENT_NAMES = {
    "vision": "Nemotron Vision Agent",
    "reasoning": "Nemotron Reasoning Agent",
    "summarize": "Llama Fast Agent",
}


@dataclass
class RoutingDecision:
    """Returned by NIMRouter.route(); carries model info + a human-readable log line."""
    task_type: str
    model_id: str
    agent_name: str
    description: str
    reason: str
    log_line: str  # For the live terminal panel


class NIMRouter:
    """
    Selects the correct NVIDIA NIM model for a given task type.

    Usage:
        router = NIMRouter()
        decision = router.route("vision", "Factory floor scene analysis")
        client  = router.get_client()
    """

    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY", "")
        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._client: Optional[OpenAI] = None

    def get_client(self) -> OpenAI:
        """Return (or lazily create) the OpenAI client pointed at NVIDIA NIM."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "NVIDIA_API_KEY not set. Add it to your .env file."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def route(self, task_type: str, context: str = "") -> RoutingDecision:
        """
        Route a task to the appropriate model.

        Args:
            task_type: One of "vision", "reasoning", "summarize"
            context:   Optional context string for the routing log

        Returns:
            RoutingDecision dataclass
        """
        task_type = task_type.lower().strip()

        if task_type not in MODELS:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Must be one of: {list(MODELS.keys())}"
            )

        model_id = MODELS[task_type]
        agent_name = AGENT_NAMES[task_type]
        description = MODEL_DESCRIPTIONS[task_type]

        reason = self._build_reason(task_type, context)
        log_line = (
            f"[NIM ROUTER] Routing... → {agent_name} selected\n"
            f"  Model   : {model_id}\n"
            f"  Reason  : {reason}"
        )

        return RoutingDecision(
            task_type=task_type,
            model_id=model_id,
            agent_name=agent_name,
            description=description,
            reason=reason,
            log_line=log_line,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_reason(self, task_type: str, context: str) -> str:
        """Produce a human-readable routing reason for the terminal panel."""
        reasons = {
            "vision": (
                "Physical scene task — Nemotron Vision specialises in spatial "
                "understanding, object state analysis, and cause-effect reasoning about the physical world."
            ),
            "reasoning": (
                "Deep analysis task — Nemotron-70B provides multi-step Chain-of-Thought "
                "reasoning, risk calculation, and structured decision-making at high accuracy."
            ),
            "summarize": (
                "Summarization task — Llama-3.1-8B delivers fast, coherent "
                "executive report generation with low latency on short contexts."
            ),
        }
        return reasons.get(task_type, "Unknown task type.")

    def get_model(self, task_type: str) -> str:
        """Return just the model ID string for a task type."""
        return MODELS.get(task_type, MODELS["summarize"])

    def list_models(self) -> dict:
        """Return the full model registry."""
        return dict(MODELS)

    def validate_connection(self) -> Tuple[bool, str]:
        """
        Ping the NVIDIA NIM endpoint with a minimal request to verify the API key.
        Returns (success: bool, message: str).
        """
        try:
            client = self.get_client()
            # Minimal token request to validate auth
            resp = client.chat.completions.create(
                model=MODELS["summarize"],
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True, f"Connection OK — model: {resp.model}"
        except Exception as exc:
            return False, f"Connection FAILED: {exc}"
