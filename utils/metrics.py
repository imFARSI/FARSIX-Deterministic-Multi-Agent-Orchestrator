"""
FARSIX Metrics — Live CPU, token, and API call tracking.

Provides:
  - CPU usage % (psutil)
  - Total tokens used (accumulated)
  - Total API calls made
  - Estimated API cost
  - Active agent count
  - Live Plotly gauge charts

Thread-safe counter accumulation.
"""

import threading
import time
from typing import Dict, Optional

import psutil


# ------------------------------------------------------------------ #
#  Token + API counters                                                 #
# ------------------------------------------------------------------ #

class MetricsCollector:
    """
    Thread-safe counter for FARSIX operational metrics.

    Usage:
        m = get_metrics()
        m.add_tokens(1247)
        m.add_api_call("vision_agent")
        cpu = m.cpu_percent()
        chart = m.cpu_gauge_figure()
    """

    # Rough cost estimates per 1k tokens (USD) for NVIDIA NIM free tier
    COST_PER_1K_TOKENS = {
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1":         0.00035,
        "nvidia/llama-3.1-nemotron-70b-instruct":  0.00080,
        "meta/llama-3.1-8b-instruct":       0.00010,
    }
    DEFAULT_COST_PER_1K = 0.00100

    def __init__(self):
        self._lock = threading.Lock()
        self._total_tokens: int = 0
        self._total_api_calls: int = 0
        self._api_calls_by_agent: Dict[str, int] = {}
        self._tokens_by_model: Dict[str, int] = {}
        self._start_time: float = time.time()
        self._cpu_samples: list = []
        self._active_agents: int = 0

    # ------------------------------------------------------------------ #
    #  Counters                                                             #
    # ------------------------------------------------------------------ #

    def add_tokens(self, count: int, model: Optional[str] = None) -> None:
        with self._lock:
            self._total_tokens += count
            if model:
                self._tokens_by_model[model] = (
                    self._tokens_by_model.get(model, 0) + count
                )

    def add_api_call(self, agent_name: str) -> None:
        with self._lock:
            self._total_api_calls += 1
            self._api_calls_by_agent[agent_name] = (
                self._api_calls_by_agent.get(agent_name, 0) + 1
            )

    def set_active_agents(self, count: int) -> None:
        with self._lock:
            self._active_agents = count

    # ------------------------------------------------------------------ #
    #  Reads                                                                #
    # ------------------------------------------------------------------ #

    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    def total_api_calls(self) -> int:
        with self._lock:
            return self._total_api_calls

    def active_agents(self) -> int:
        with self._lock:
            return self._active_agents

    def estimated_cost_usd(self) -> float:
        """Estimate total USD cost based on token usage per model."""
        with self._lock:
            total = 0.0
            for model, tokens in self._tokens_by_model.items():
                rate = self.COST_PER_1K_TOKENS.get(model, self.DEFAULT_COST_PER_1K)
                total += (tokens / 1000) * rate
            # If no model breakdown, use default rate
            if not self._tokens_by_model and self._total_tokens > 0:
                total = (self._total_tokens / 1000) * self.DEFAULT_COST_PER_1K
        return round(total, 6)

    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 1)

    # ------------------------------------------------------------------ #
    #  CPU                                                                  #
    # ------------------------------------------------------------------ #

    def cpu_percent(self) -> float:
        """Return current CPU usage percentage (non-blocking)."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def memory_percent(self) -> float:
        """Return current RAM usage percentage."""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    def memory_used_gb(self) -> float:
        """Return used RAM in GB."""
        try:
            vm = psutil.virtual_memory()
            return round(vm.used / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def memory_total_gb(self) -> float:
        """Return total RAM in GB."""
        try:
            vm = psutil.virtual_memory()
            return round(vm.total / (1024 ** 3), 2)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  Plotly charts                                                        #
    # ------------------------------------------------------------------ #

    def cpu_gauge_figure(self):
        """Return a Plotly gauge figure for CPU usage."""
        import plotly.graph_objects as go

        cpu = self.cpu_percent()

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu,
            number={"suffix": "%", "font": {"size": 28, "color": "#E8EAF6"}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "CPU Usage", "font": {"size": 14, "color": "#90A4AE"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#455A64",
                    "tickfont": {"color": "#90A4AE"},
                },
                "bar": {"color": _cpu_color(cpu)},
                "bgcolor": "#1C2833",
                "borderwidth": 2,
                "bordercolor": "#455A64",
                "steps": [
                    {"range": [0, 40],  "color": "#1B5E20"},
                    {"range": [40, 70], "color": "#E65100"},
                    {"range": [70, 100],"color": "#B71C1C"},
                ],
                "threshold": {
                    "line": {"color": "#FFD54F", "width": 3},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
            height=200,
        )
        return fig

    def memory_gauge_figure(self):
        """Return a Plotly gauge for RAM usage."""
        import plotly.graph_objects as go

        used = self.memory_used_gb()
        total = self.memory_total_gb()
        pct = self.memory_percent()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=used,
            number={"suffix": " GB", "font": {"size": 22, "color": "#E8EAF6"}},
            delta={"reference": total, "valueformat": ".1f",
                   "suffix": " GB total", "font": {"size": 12}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"RAM Usage ({pct:.0f}%)",
                   "font": {"size": 14, "color": "#90A4AE"}},
            gauge={
                "axis": {"range": [0, total]},
                "bar": {"color": "#4FC3F7"},
                "bgcolor": "#1C2833",
                "borderwidth": 2,
                "bordercolor": "#455A64",
                "steps": [
                    {"range": [0, total * 0.6],  "color": "#1B5E20"},
                    {"range": [total * 0.6, total * 0.8], "color": "#E65100"},
                    {"range": [total * 0.8, total], "color": "#B71C1C"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
            height=200,
        )
        return fig

    def summary_dict(self) -> dict:
        """Return all metrics as a dict for the dashboard top bar."""
        return {
            "cpu_pct":        self.cpu_percent(),
            "memory_pct":     self.memory_percent(),
            "memory_used_gb": self.memory_used_gb(),
            "memory_total_gb":self.memory_total_gb(),
            "total_tokens":   self.total_tokens(),
            "total_api_calls":self.total_api_calls(),
            "active_agents":  self.active_agents(),
            "estimated_cost": self.estimated_cost_usd(),
            "uptime_s":       self.uptime_seconds(),
        }

    def reset(self) -> None:
        with self._lock:
            self._total_tokens = 0
            self._total_api_calls = 0
            self._api_calls_by_agent.clear()
            self._tokens_by_model.clear()


def _cpu_color(cpu_pct: float) -> str:
    if cpu_pct < 40:
        return "#4CAF50"
    elif cpu_pct < 70:
        return "#FF9800"
    else:
        return "#F44336"


# ------------------------------------------------------------------ #
#  Singleton                                                            #
# ------------------------------------------------------------------ #

_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
