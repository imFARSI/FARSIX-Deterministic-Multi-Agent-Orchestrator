"""
FARSIX Recovery Engine — Handles agent timeouts, crash detection, and auto-restart.
"""

from __future__ import annotations

import asyncio
import functools
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, Optional

from backend.event_bus import Event, EventBus, get_event_bus

# Default values
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 2


class AgentTimeoutError(Exception):
    """Raised when an agent coroutine exceeds the timeout."""


class AgentCrashError(Exception):
    """Raised when an agent coroutine raises an unexpected exception."""


class RecoveryEngine:
    """
    Wraps async agent calls with timeout, retry, and event-publishing logic.

    Usage:
        engine = RecoveryEngine(event_bus)
        result = await engine.run_with_recovery(
            coro=my_agent.run(input_data),
            agent_name="vision_agent",
            mission_id="m-001",
            timeout=45,
            max_retries=2,
        )
    """

    def __init__(self, event_bus: Optional[EventBus] = None,
                 timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.bus = event_bus or get_event_bus()
        self.default_timeout = timeout
        self.default_max_retries = max_retries

        # Per-agent state tracking
        self._agent_status: Dict[str, str] = {}   # agent_name → "ONLINE" | "OFFLINE"
        self._agent_retries: Dict[str, int] = {}  # agent_name → retry_count

    # ------------------------------------------------------------------ #
    #  Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def run_with_recovery(
        self,
        coro_factory: Callable[[], Coroutine],
        agent_name: str,
        mission_id: str,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Any:
        """
        Execute coro_factory() with timeout and retry logic.

        Args:
            coro_factory: Zero-arg callable that returns a fresh coroutine each time.
                          (A new coroutine is needed for each retry attempt.)
            agent_name:   Name used in event bus messages.
            mission_id:   Mission this agent call belongs to.
            timeout:      Override default timeout (seconds).
            max_retries:  Override default max retries.

        Returns:
            The coroutine's return value on success.

        Raises:
            AgentTimeoutError | AgentCrashError if all retries are exhausted.
        """
        t_out = timeout if timeout is not None else self.default_timeout
        retries = max_retries if max_retries is not None else self.default_max_retries

        last_exception: Optional[Exception] = None
        self._agent_status[agent_name] = "ONLINE"
        self._agent_retries[agent_name] = 0

        for attempt in range(retries + 1):
            if attempt > 0:
                self._publish_retry(agent_name, mission_id, attempt, retries, last_exception)
                await asyncio.sleep(1.5)  # Brief pause before retry

            try:
                coro = coro_factory()
                result = await asyncio.wait_for(coro, timeout=t_out)
                self._agent_status[agent_name] = "ONLINE"
                self._agent_retries[agent_name] = attempt
                self._publish_recovery_ok(agent_name, mission_id, attempt)
                return result

            except asyncio.TimeoutError:
                last_exception = AgentTimeoutError(
                    f"{agent_name} timed out after {t_out}s (attempt {attempt + 1}/{retries + 1})"
                )
                self._mark_offline(agent_name, mission_id, str(last_exception))

            except AgentTimeoutError as exc:
                last_exception = exc
                self._mark_offline(agent_name, mission_id, str(exc))

            except Exception as exc:
                tb = traceback.format_exc()
                last_exception = AgentCrashError(
                    f"{agent_name} crashed (attempt {attempt + 1}): {exc}\n{tb}"
                )
                self._mark_offline(agent_name, mission_id, str(last_exception))

        # All retries exhausted
        self._publish_failed(agent_name, mission_id, str(last_exception))
        raise last_exception  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    #  Status helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_agent_status(self, agent_name: str) -> str:
        return self._agent_status.get(agent_name, "UNKNOWN")

    def all_statuses(self) -> Dict[str, str]:
        return dict(self._agent_status)

    def mark_idle(self, agent_name: str) -> None:
        self._agent_status[agent_name] = "IDLE"

    # ------------------------------------------------------------------ #
    #  Internal event publishers                                            #
    # ------------------------------------------------------------------ #

    def _mark_offline(self, agent_name: str, mission_id: str, reason: str) -> None:
        self._agent_status[agent_name] = "OFFLINE"
        self.bus.publish(Event(
            event_type="agent_offline",
            source=agent_name,
            mission_id=mission_id,
            data={"reason": reason, "status": "OFFLINE"},
        ))

    def _publish_retry(self, agent_name: str, mission_id: str,
                       attempt: int, max_retries: int,
                       exc: Optional[Exception]) -> None:
        self.bus.publish(Event(
            event_type="agent_retry",
            source=agent_name,
            mission_id=mission_id,
            data={
                "attempt": attempt,
                "max_retries": max_retries,
                "reason": str(exc),
                "message": (
                    f"⟳ RETRY {attempt}/{max_retries} — {agent_name.upper()} "
                    f"restarting after error..."
                ),
            },
        ))

    def _publish_recovery_ok(self, agent_name: str, mission_id: str,
                              attempt: int) -> None:
        if attempt > 0:
            self.bus.publish(Event(
                event_type="agent_recovered",
                source=agent_name,
                mission_id=mission_id,
                data={
                    "attempt": attempt,
                    "message": f"✅ {agent_name.upper()} recovered on attempt {attempt + 1}",
                },
            ))

    def _publish_failed(self, agent_name: str, mission_id: str,
                        reason: str) -> None:
        self.bus.publish(Event(
            event_type="agent_failed",
            source=agent_name,
            mission_id=mission_id,
            data={
                "reason": reason,
                "message": (
                    f"⚠️ {agent_name.upper()} FAILED after all retries. "
                    "Mission marked FAILED."
                ),
            },
        ))


# ------------------------------------------------------------------ #
#  Decorator helper                                                     #
# ------------------------------------------------------------------ #

def with_recovery(agent_name: str, mission_id_kwarg: str = "mission_id",
                  timeout: int = DEFAULT_TIMEOUT_SECONDS,
                  max_retries: int = DEFAULT_MAX_RETRIES):
    """
    Decorator that wraps an async method with recovery logic.

    The decorated method must accept a `mission_id` keyword argument.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            mid = kwargs.get(mission_id_kwarg, "unknown")
            engine = RecoveryEngine()
            return await engine.run_with_recovery(
                coro_factory=lambda: func(*args, **kwargs),
                agent_name=agent_name,
                mission_id=mid,
                timeout=timeout,
                max_retries=max_retries,
            )
        return wrapper
    return decorator


# Singleton
_recovery_engine: Optional[RecoveryEngine] = None


def get_recovery_engine() -> RecoveryEngine:
    global _recovery_engine
    if _recovery_engine is None:
        _recovery_engine = RecoveryEngine()
    return _recovery_engine
