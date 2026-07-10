"""
FARSIX Event Bus — Local async publish/subscribe event system.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Event:
    """A single event on the FARSIX event bus."""
    event_type: str          # e.g. "agent_started", "agent_complete", "mission_state_change"
    source: str              # e.g. "vision_agent", "mission_engine"
    mission_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "mission_id": self.mission_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class EventBus:
    """
    Thread-safe, async-compatible event bus.
    Agents publish events; the UI and mission engine subscribe via callbacks
    or by polling the shared deque.

    Usage:
        bus = EventBus()
        bus.subscribe("agent_complete", my_callback)
        bus.publish(Event("agent_complete", "vision_agent", "mission-001", {"result": ...}))
    """

    def __init__(self, max_history: int = 500):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Subscription                                                         #
    # ------------------------------------------------------------------ #

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a callback for a specific event type. Use '*' for all events."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    # ------------------------------------------------------------------ #
    #  Publishing                                                           #
    # ------------------------------------------------------------------ #

    def publish(self, event: Event) -> None:
        """Publish an event synchronously. Safe to call from any thread."""
        with self._lock:
            self._history.append(event)
            callbacks = list(self._subscribers.get(event.event_type, []))
            callbacks += list(self._subscribers.get("*", []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Never let a subscriber crash the bus

    async def publish_async(self, event: Event) -> None:
        """Async publish — runs callbacks in a thread executor to avoid blocking the loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.publish, event)

    # ------------------------------------------------------------------ #
    #  History access (for UI polling)                                      #
    # ------------------------------------------------------------------ #

    def get_history(self, limit: Optional[int] = None) -> List[Event]:
        """Return event history, newest last."""
        with self._lock:
            events = list(self._history)
        return events[-limit:] if limit else events

    def get_mission_events(self, mission_id: str) -> List[Event]:
        """Return all events for a specific mission."""
        with self._lock:
            return [e for e in self._history if e.mission_id == mission_id]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------ #
    #  Convenience factory                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def make(event_type: str, source: str, mission_id: str, **kwargs) -> Event:
        """Shorthand for creating an event with keyword data."""
        return Event(event_type=event_type, source=source, mission_id=mission_id, data=dict(kwargs))


# Singleton instance shared across the whole process
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return the process-wide singleton event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus(max_history=500)
    return _global_bus
