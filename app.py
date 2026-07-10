"""
FARSIX — Framework for Agentic Reasoning and Supervised Intelligence (X-tended)
Main Streamlit Dashboard

Single entry point: streamlit run app.py

Built by Salman Farsi — Undergraduate AI Researcher, Brac University
"""

from __future__ import annotations

import asyncio
import datetime
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# Fix asyncio on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FARSIX — Multi-Agent Physical AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root ─────────────────────────────────────────────── */
:root {
  --bg-primary:   #0D1117;
  --bg-secondary: #161B22;
  --bg-card:      #1C2333;
  --bg-card2:     #21262D;
  --border:       #30363D;
  --text-primary: #E6EDF3;
  --text-muted:   #8B949E;
  --accent-blue:  #4FC3F7;
  --accent-purple:#CE93D8;
  --accent-green: #56D364;
  --accent-yellow:#FFD54F;
  --accent-red:   #F85149;
  --accent-orange:#FF9800;
  --glow-blue:    0 0 20px rgba(79,195,247,0.35);
  --glow-purple:  0 0 20px rgba(206,147,216,0.35);
  --glow-green:   0 0 20px rgba(86,211,100,0.35);
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; }

/* ── Hide Streamlit branding ──────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Containers ──────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 12px;
}
.card-glow-blue  { box-shadow: var(--glow-blue);  }
.card-glow-green { box-shadow: var(--glow-green); }

/* ── Top bar ─────────────────────────────────────────── */
.topbar {
  background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.topbar-title {
  font-size: 1.55rem;
  font-weight: 700;
  background: linear-gradient(90deg, #4FC3F7, #CE93D8, #FFD54F);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 0.5px;
}
.topbar-subtitle {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 2px;
  letter-spacing: 0.3px;
}
.metric-chip {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 14px;
  font-size: 0.78rem;
  font-family: 'JetBrains Mono', monospace;
  display: inline-block;
  margin: 0 4px;
  color: var(--text-primary);
}
.metric-chip-green  { border-color: var(--accent-green);  color: var(--accent-green); }
.metric-chip-blue   { border-color: var(--accent-blue);   color: var(--accent-blue); }
.metric-chip-purple { border-color: var(--accent-purple); color: var(--accent-purple); }
.metric-chip-yellow { border-color: var(--accent-yellow); color: var(--accent-yellow); }

/* ── Section headers ─────────────────────────────────── */
.section-header {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── Mission badge ───────────────────────────────────── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.68rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.5px;
}
.badge-queued     { background: #1C2333; color: #8B949E; border: 1px solid #30363D; }
.badge-running    { background: #0D2137; color: #4FC3F7; border: 1px solid #4FC3F7; animation: pulse-blue 1.5s infinite; }
.badge-complete   { background: #0D2B0D; color: #56D364; border: 1px solid #56D364; }
.badge-failed     { background: #2D0D0D; color: #F85149; border: 1px solid #F85149; }
.badge-parsing    { background: #1A1A00; color: #FFD54F; border: 1px solid #FFD54F; }
.badge-vision     { background: #0D1A2D; color: #4FC3F7; border: 1px solid #4FC3F7; animation: pulse-blue 1.5s infinite; }
.badge-reasoning  { background: #1A0D2D; color: #CE93D8; border: 1px solid #CE93D8; animation: pulse-purple 1.5s infinite; }
.badge-validation { background: #2D0D0D; color: #FF9800; border: 1px solid #FF9800; }
.badge-summarizing{ background: #0D2B0D; color: #A5D6A7; border: 1px solid #A5D6A7; }

@keyframes pulse-blue {
  0%, 100% { box-shadow: 0 0 0 0 rgba(79,195,247,0.4); }
  50%       { box-shadow: 0 0 8px 3px rgba(79,195,247,0.4); }
}
@keyframes pulse-purple {
  0%, 100% { box-shadow: 0 0 0 0 rgba(206,147,216,0.4); }
  50%       { box-shadow: 0 0 8px 3px rgba(206,147,216,0.4); }
}

/* ── Terminal log ────────────────────────────────────── */
.terminal {
  background: #050810;
  border: 1px solid #1E3A5F;
  border-radius: 10px;
  padding: 14px 16px;
  height: 360px;
  overflow-y: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  line-height: 1.7;
}
.terminal::-webkit-scrollbar { width: 4px; }
.terminal::-webkit-scrollbar-track { background: #050810; }
.terminal::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 2px; }

.log-vision     { color: #4FC3F7; }
.log-nemotron   { color: #CE93D8; }
.log-llama      { color: #A5D6A7; }
.log-guardrails-ok   { color: #56D364; }
.log-guardrails-fail { color: #F85149; }
.log-router     { color: #FFD54F; }
.log-engine     { color: #8B949E; }
.log-memory     { color: #FF9800; }
.log-system     { color: #546E7A; }

/* ── Progress bar ────────────────────────────────────── */
.progress-bar-outer {
  background: var(--bg-card2);
  border-radius: 6px;
  height: 8px;
  width: 100%;
  margin: 6px 0;
  overflow: hidden;
}
.progress-bar-inner {
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, #4FC3F7, #CE93D8);
  transition: width 0.5s ease;
}

/* ── Skill notification ──────────────────────────────── */
.skill-notification {
  background: linear-gradient(135deg, #0D2B0D, #1B5E20);
  border: 1px solid #56D364;
  border-radius: 10px;
  padding: 10px 16px;
  animation: skill-pop 0.5s ease;
  margin: 8px 0;
}
@keyframes skill-pop {
  0%   { transform: scale(0.95); opacity: 0; }
  100% { transform: scale(1);    opacity: 1; }
}

/* ── Timeline table ──────────────────────────────────── */
.timeline-row {
  display: grid;
  grid-template-columns: 110px 90px 130px 130px 1fr;
  gap: 8px;
  padding: 5px 10px;
  font-size: 0.7rem;
  font-family: 'JetBrains Mono', monospace;
  border-bottom: 1px solid #161B22;
}
.timeline-header {
  font-weight: 600;
  color: var(--text-muted);
  border-bottom: 2px solid var(--border) !important;
  padding-bottom: 6px !important;
}
.timeline-body { max-height: 220px; overflow-y: auto; }

/* ── Input type tabs ─────────────────────────────────── */
.stRadio > div { flex-direction: row !important; flex-wrap: wrap; gap: 8px; }
.stRadio label {
  background: var(--bg-card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 8px 18px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  color: var(--text-primary) !important;
}
.stRadio label:hover { border-color: var(--accent-blue) !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: 10px 20px !important;
  transition: all 0.2s !important;
  width: 100% !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #1976D2, #1565C0) !important;
  box-shadow: 0 4px 16px rgba(79,195,247,0.4) !important;
  transform: translateY(-1px) !important;
}

/* ── Selectbox / text area ────────────────────────────── */
.stSelectbox > div > div, .stTextArea > div > div {
  background: var(--bg-card2) !important;
  border-color: var(--border) !important;
  color: var(--text-primary) !important;
}
textarea { color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; }

/* ── Demo buttons ────────────────────────────────────── */
.demo-btn { margin: 4px 0; }

/* ── Metric cards ────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 12px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Upload ──────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg-card2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}

/* ── DataFrame ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "missions":            [],   # list of mission dicts
        "log_lines":           [],   # terminal log entries
        "agent_states":        {     # node → state string
            "nim_router":       "IDLE",
            "vision_agent":     "IDLE",
            "nemotron_agent":   "IDLE",
            "guardrails_agent": "IDLE",
            "llama_agent":      "IDLE",
        },
        "active_edges":        [],
        "new_skill":           None,  # str notification
        "running_mission_id":  None,
        "total_tokens":        0,
        "total_api_calls":     0,
        "engine_initialised":  False,
        "last_refresh":        0.0,
        "event_bus_processed": 0,   # index into event bus history
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()




# ─────────────────────────────────────────────────────────────────────────────
# Backend accessors (lazy — only import when needed)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_engine():
    from backend.mission_engine import get_mission_engine
    return get_mission_engine()

@st.cache_resource
def _get_metrics():
    from utils.metrics import get_metrics
    return get_metrics()

@st.cache_resource
def _get_bus():
    from backend.event_bus import get_event_bus
    return get_event_bus()

@st.cache_resource
def _get_skill_lib():
    from backend.skill_library import get_skill_library
    return get_skill_library()

@st.cache_resource
def _get_chroma():
    try:
        from memory.chroma_store import get_chroma_store
        return get_chroma_store()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Event bus → session state sync
# ─────────────────────────────────────────────────────────────────────────────

AGENT_LOG_COLORS = {
    "vision_agent":     "log-vision",
    "nemotron_agent":   "log-nemotron",
    "llama_agent":      "log-llama",
    "guardrails_agent": "log-guardrails-ok",
    "mission_engine":   "log-engine",
    "nim_router":       "log-router",
}

STATE_TRANSITIONS = {
    "VISION_ANALYSIS": ("nim_router", "vision_agent", "RUNNING"),
    "DEEP_REASONING":  ("vision_agent", "nemotron_agent", "RUNNING"),
    "VALIDATION":      ("nemotron_agent", "guardrails_agent", "RUNNING"),
    "SUMMARIZING":     ("guardrails_agent", "llama_agent", "RUNNING"),
    "COMPLETE":        (None, None, "COMPLETE"),
    "FAILED":          (None, None, "FAILED"),
}


def _sync_events():
    """Pull new events from the event bus into session state."""
    bus = _get_bus()
    history = bus.get_history()
    already = st.session_state.event_bus_processed
    new_events = history[already:]
    st.session_state.event_bus_processed = len(history)

    for event in new_events:
        _process_event(event)


def _process_event(event):
    """Update session state based on an event."""
    data = event.data or {}
    source = event.source
    etype = event.event_type
    ts = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")

    # ── Terminal log ──────────────────────────────────────────────────
    msg = data.get("message", "")
    if msg:
        css = AGENT_LOG_COLORS.get(source, "log-system")
        if "guardrails" in source:
            css = "log-guardrails-ok" if "APPROVED" in msg or "✅" in msg else "log-guardrails-fail"
        prefix = {
            "vision_agent":     "[NEMOTRON VISION]     ",
            "nemotron_agent":   "[NEMOTRON]   ",
            "llama_agent":      "[LLAMA]      ",
            "guardrails_agent": "[GUARDRAILS] ",
            "mission_engine":   "[ENGINE]     ",
            "nim_router":       "[NIM ROUTER] ",
        }.get(source, f"[{source.upper()}]")

        for line in msg.strip().split("\n"):
            if line.strip():
                st.session_state.log_lines.append(
                    f'<span class="{css}">{ts} {prefix} {line.strip()}</span>'
                )
        # Cap log at 300 entries
        st.session_state.log_lines = st.session_state.log_lines[-300:]

    # ── Agent states ──────────────────────────────────────────────────
    if etype == "agent_started":
        st.session_state.agent_states[source] = "RUNNING"
    elif etype in ("agent_complete", "agent_recovered", "guardrails_approved"):
        st.session_state.agent_states[source] = "COMPLETE"
    elif etype in ("agent_offline", "agent_failed", "guardrails_blocked"):
        st.session_state.agent_states[source] = "OFFLINE"

    # ── State machine transitions ─────────────────────────────────────
    if etype == "mission_state_change":
        new_state = data.get("new_state", "")
        transition = STATE_TRANSITIONS.get(new_state)
        if transition:
            src_node, tgt_node, tgt_state = transition
            if tgt_node:
                st.session_state.agent_states[tgt_node] = tgt_state
                st.session_state.active_edges = [(src_node, tgt_node)] if src_node else []
        if new_state in ("COMPLETE", "FAILED"):
            st.session_state.agent_states = {k: "IDLE" for k in st.session_state.agent_states}
            st.session_state.active_edges = []
            st.session_state.running_mission_id = None

    # ── Token counter ─────────────────────────────────────────────────
    tokens = data.get("tokens_used", 0)
    if tokens:
        st.session_state.total_tokens += tokens
        _get_metrics().add_tokens(tokens)

    if etype in ("agent_started",) and source in ("vision_agent", "nemotron_agent", "llama_agent"):
        st.session_state.total_api_calls += 1
        _get_metrics().add_api_call(source)

    # ── New skill notification ────────────────────────────────────────
    new_skill_msg = data.get("new_skill_notification")
    if new_skill_msg:
        st.session_state.new_skill = new_skill_msg

    # ── Mission list update ────────────────────────────────────────────
    engine = _get_engine()
    st.session_state.missions = [m.to_dict() for m in engine.list_missions()]


# ─────────────────────────────────────────────────────────────────────────────
# Mission runner (background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_mission_async(mission_id: str):
    """Run a mission in a background thread with its own event loop."""
    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            engine = _get_engine()
            loop.run_until_complete(engine.run_mission(mission_id))
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _badge(state: str) -> str:
    state_upper = state.upper()
    css_map = {
        "QUEUED":          "badge-queued",
        "PARSING_INPUT":   "badge-parsing",
        "VISION_ANALYSIS": "badge-vision",
        "DEEP_REASONING":  "badge-reasoning",
        "VALIDATION":      "badge-validation",
        "SUMMARIZING":     "badge-summarizing",
        "COMPLETE":        "badge-complete",
        "FAILED":          "badge-failed",
    }
    css = css_map.get(state_upper, "badge-queued")
    return f'<span class="badge {css}">{state_upper}</span>'


def _progress_bar(pct: int) -> str:
    return (
        f'<div class="progress-bar-outer">'
        f'<div class="progress-bar-inner" style="width:{pct}%"></div>'
        f'</div>'
    )


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Input parser bridge
# ─────────────────────────────────────────────────────────────────────────────

def _parse_input(input_type: str, text_content: str,
                 uploaded_file) -> Tuple[str, str]:
    """Bridge to utils/input_parser.py."""
    from utils.input_parser import parse_input
    file_bytes = uploaded_file.read() if uploaded_file else None
    filename = uploaded_file.name if uploaded_file else "upload"
    return parse_input(
        input_type=input_type,
        text_content=text_content,
        file_bytes=file_bytes,
        filename=filename,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_topbar():
    _sync_events()
    metrics = _get_metrics()
    cpu = metrics.cpu_percent()
    active = st.session_state.agent_states
    running_count = sum(1 for v in active.values() if v == "RUNNING")
    mission_count = len(st.session_state.missions)
    tokens = st.session_state.total_tokens
    api_calls = st.session_state.total_api_calls

    cpu_color = "#4CAF50" if cpu < 40 else "#FF9800" if cpu < 70 else "#F44336"

    st.markdown(f"""
    <div class="topbar">
      <div>
        <div class="topbar-title">⚡ FARSIX</div>
        <div class="topbar-subtitle">Framework for Agentic Reasoning and Supervised Intelligence (X-tended)</div>
      </div>
      <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
        <span class="metric-chip metric-chip-green" style="color:{cpu_color};border-color:{cpu_color};">
          🖥️ CPU: {cpu:.0f}%
        </span>
        <span class="metric-chip metric-chip-blue">
          🤖 Agents: {running_count} active
        </span>
        <span class="metric-chip metric-chip-purple">
          🔤 Tokens: {tokens:,}
        </span>
        <span class="metric-chip metric-chip-yellow">
          📡 API Calls: {api_calls}
        </span>
        <span class="metric-chip">
          🚀 Missions: {mission_count}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Mission Control
# ─────────────────────────────────────────────────────────────────────────────

def render_mission_control():
    st.markdown('<div class="section-header">🎯 Mission Control</div>', unsafe_allow_html=True)



    # ── Input type ────────────────────────────────────────────────────────
    st.markdown("**Input Mode**")
    input_type = st.radio(
        "input_mode",
        options=["Text", "PDF", "CSV", "Image"],
        horizontal=True,
        label_visibility="collapsed",
    ).lower()

    # ── File upload / text ────────────────────────────────────────────────
    uploaded_file = None
    text_content = ""

    if input_type == "text":
        text_content = st.text_area(
            "Scene Description",
            placeholder="Describe the physical scene, sensor readings, or situation to analyse...",
            height=150,
            label_visibility="collapsed",
        )
    elif input_type == "pdf":
        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"], label_visibility="collapsed"
        )
    elif input_type == "csv":
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed"
        )
    elif input_type == "image":
        uploaded_file = st.file_uploader(
            "Upload Image", type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded scene", use_container_width=True)

    # ── Mission goal ──────────────────────────────────────────────────────
    goal = st.text_area(
        "Mission Goal",
        placeholder="What should the agents analyse and produce?",
        height=90,
    )

    # ── Submit ────────────────────────────────────────────────────────────
    can_submit = (
        goal.strip() and
        (text_content.strip() or uploaded_file is not None) and
        st.session_state.running_mission_id is None
    )

    if st.button(
        "🚀 Launch Mission" if can_submit else "⏳ Mission Running..." if st.session_state.running_mission_id else "🚀 Launch Mission",
        disabled=not can_submit,
        use_container_width=True,
        key="launch_btn",
    ):
        _launch_mission(input_type, text_content, goal, uploaded_file)
        st.rerun()

    if st.session_state.running_mission_id:
        st.markdown(
            '<div style="text-align:center;color:#4FC3F7;font-size:0.8rem;margin-top:4px;">'
            '⟳ Mission in progress — auto-refreshing...</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Mission queue ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Mission Queue</div>', unsafe_allow_html=True)
    missions = st.session_state.missions

    if not missions:
        st.markdown(
            '<div style="color:#546E7A;font-size:0.8rem;text-align:center;padding:20px 0;">'
            'No missions yet. Launch one above!</div>',
            unsafe_allow_html=True,
        )
    else:
        for m in missions[:8]:
            state = m.get("state", "QUEUED")
            elapsed = m.get("elapsed", 0)
            pct = m.get("progress_pct", 0)
            goal_text = m.get("goal", "")[:55] + ("..." if len(m.get("goal", "")) > 55 else "")
            retries = m.get("retry_count", 0)
            retry_str = f" ↺{retries}" if retries > 0 else ""

            st.markdown(
                f'<div class="card" style="padding:10px 14px;margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:0.72rem;color:#8B949E;">{m["id"]}{retry_str}</span>'
                f'{_badge(state)}'
                f'</div>'
                f'<div style="font-size:0.78rem;margin:4px 0 2px 0;">{goal_text}</div>'
                f'{_progress_bar(pct)}'
                f'<div style="font-size:0.68rem;color:#546E7A;">⏱ {_format_elapsed(elapsed)} | {m["input_type"].upper()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _launch_mission(input_type: str, text_content: str, goal: str, uploaded_file):
    """Parse input, create mission, and launch in background."""
    extracted_text, _meta = _parse_input(input_type, text_content, uploaded_file)

    input_image_b64 = None
    if input_type == "image" and uploaded_file:
        import base64
        uploaded_file.seek(0)
        input_image_b64 = base64.b64encode(uploaded_file.read()).decode('utf-8')

    engine = _get_engine()
    mission = engine.create_mission(
        goal=goal,
        input_type=input_type,
        input_text=extracted_text,
        input_image_b64=input_image_b64,
    )
    st.session_state.running_mission_id = mission.id
    st.session_state.missions = [m.to_dict() for m in engine.list_missions()]
    st.session_state.log_lines.append(
        f'<span class="log-engine">{datetime.datetime.now().strftime("%H:%M:%S")} '
        f'[ENGINE]     🚀 Mission {mission.id} created — launching agents...</span>'
    )
    _run_mission_async(mission.id)


# ─────────────────────────────────────────────────────────────────────────────
# CENTER PANEL — Live Agent Graph
# ─────────────────────────────────────────────────────────────────────────────

def render_agent_graph():
    st.markdown('<div class="section-header">🕸️ Live Agent Graph</div>', unsafe_allow_html=True)

    try:
        from utils.graph_viz import build_agent_graph, get_state_emoji
        html = build_agent_graph(
            agent_states=st.session_state.agent_states,
            active_edges=st.session_state.active_edges,
            height=600,
        )
        import streamlit.components.v1 as components
        components.html(html, height=620, scrolling=False)
    except Exception as exc:
        # Fallback: text-based agent status
        st.markdown("**Agent Status**")
        states = st.session_state.agent_states
        emojis = {"IDLE": "🟢", "RUNNING": "🔵", "COMPLETE": "✅", "OFFLINE": "🔴", "FAILED": "⚠️"}
        names = {
            "nim_router":       "NIM Router",
            "vision_agent":     "Nemotron Vision",
            "nemotron_agent":   "Nemotron Reason",
            "guardrails_agent": "NeMo Guardrails",
            "llama_agent":      "Llama Fast",
        }
        for node_id, name in names.items():
            state = states.get(node_id, "IDLE")
            emoji = emojis.get(state.upper(), "⚪")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                f'border-bottom:1px solid #161B22;">'
                f'<span style="font-size:0.8rem;">{emoji} {name}</span>'
                f'<span style="font-size:0.72rem;color:#546E7A;">{state}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Agent state summary cards ─────────────────────────────────────────
    agent_display = {
        "vision_agent":     ("NEMOTRON VISION",      "#4FC3F7", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"),
        "nemotron_agent":   ("REASONING",    "#CE93D8", "meta/llama-3.1-70b-instruct"),
        "guardrails_agent": ("GUARDRAILS",  "#EF9A9A", "NeMo Local CPU"),
        "llama_agent":      ("LLAMA",       "#A5D6A7", "llama-3.1-8b-instruct"),
    }
    cols = st.columns(4)
    state_icons = {"IDLE": "🟢", "RUNNING": "🔵", "COMPLETE": "✅", "OFFLINE": "🔴", "FAILED": "⚠️"}
    for i, (node_id, (name, color, model)) in enumerate(agent_display.items()):
        state = st.session_state.agent_states.get(node_id, "IDLE")
        icon = state_icons.get(state.upper(), "⚪")
        with cols[i]:
            st.markdown(
                f'<div class="card" style="border-color:{color}33;padding:10px 12px;text-align:center;">'
                f'<div style="font-size:1.2rem;">{icon}</div>'
                f'<div style="font-size:0.72rem;font-weight:700;color:{color};margin:2px 0;">{name}</div>'
                f'<div style="font-size:0.62rem;color:#546E7A;">{model}</div>'
                f'<div style="font-size:0.68rem;color:#8B949E;margin-top:4px;">{state}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — Live Agent Terminal
# ─────────────────────────────────────────────────────────────────────────────

def render_terminal():
    st.markdown('<div class="section-header">💻 Live Agent Terminal</div>', unsafe_allow_html=True)

    log_lines = st.session_state.log_lines[-80:]  # Show last 80 lines
    if not log_lines:
        log_lines = [
            '<span class="log-system">08:00:00 [SYSTEM]     FARSIX initialised. Waiting for mission...</span>',
            '<span class="log-system">08:00:00 [SYSTEM]     Models: Nemotron Vision • Nemotron • Llama • NeMo Guardrails</span>',
            '<span class="log-router">08:00:00 [NIM ROUTER] Ready to route tasks to NVIDIA NIM endpoints.</span>',
        ]

    log_html = "<br>".join(log_lines)
    terminal_html = f"""
    <div class="terminal" id="farsix-terminal">
      {log_html}
    </div>
    <script>
      var t = document.getElementById('farsix-terminal');
      if(t) t.scrollTop = t.scrollHeight;
    </script>
    """
    st.markdown(terminal_html, unsafe_allow_html=True)

    # Legend
    st.markdown(
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">'
        '<span style="font-size:0.68rem;"><span style="color:#4FC3F7;">■</span> NEMOTRON VISION</span>'
        '<span style="font-size:0.68rem;"><span style="color:#CE93D8;">■</span> NEMOTRON</span>'
        '<span style="font-size:0.68rem;"><span style="color:#A5D6A7;">■</span> LLAMA</span>'
        '<span style="font-size:0.68rem;"><span style="color:#56D364;">■</span> GUARDRAILS OK</span>'
        '<span style="font-size:0.68rem;"><span style="color:#F85149;">■</span> GUARDRAILS FAIL</span>'
        '<span style="font-size:0.68rem;"><span style="color:#FFD54F;">■</span> ROUTER</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM LEFT — System Metrics
# ─────────────────────────────────────────────────────────────────────────────

def render_metrics():
    st.markdown('<div class="section-header">📊 System Metrics</div>', unsafe_allow_html=True)

    metrics = _get_metrics()
    chroma = _get_chroma()
    chroma_count = chroma.count() if chroma and chroma.is_available() else 0

    # Gauge charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            metrics.cpu_gauge_figure(),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col2:
        st.plotly_chart(
            metrics.memory_gauge_figure(),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Stat row
    running_agents = sum(
        1 for v in st.session_state.agent_states.values() if v == "RUNNING"
    )
    tokens = st.session_state.total_tokens
    api_calls = st.session_state.total_api_calls

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🤖 Active Agents", running_agents)
    with c2:
        st.metric("📡 API Calls", api_calls)
    with c3:
        st.metric("🔤 Total Tokens", f"{tokens:,}")
    with c4:
        st.metric("🧠 Memory Entries", chroma_count)




# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM RIGHT — Skill Library
# ─────────────────────────────────────────────────────────────────────────────

def render_skill_library():
    st.markdown('<div class="section-header">📚 Skill Library</div>', unsafe_allow_html=True)

    # New skill notification
    new_skill = st.session_state.new_skill
    if new_skill:
        st.markdown(
            f'<div class="skill-notification">'
            f'<span style="color:#56D364;font-weight:700;font-size:0.85rem;">{new_skill}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Auto-clear after showing
        st.session_state.new_skill = None

    skill_lib = _get_skill_lib()
    skills = skill_lib.get_all_skills()

    if not skills:
        st.markdown(
            '<div style="color:#546E7A;font-size:0.8rem;text-align:center;padding:20px 0;">'
            'No skills yet. Complete a mission to learn a new skill!</div>',
            unsafe_allow_html=True,
        )
        return

    import pandas as pd
    skill_data = []
    for s in skills:
        skill_data.append({
            "Name": s.name[:30],
            "Type": s.input_type.upper(),
            "Uses": s.usage_count,
            "Success": f"{s.success_rate * 100:.0f}%",
            "Created": datetime.datetime.fromtimestamp(s.created_at).strftime("%m/%d %H:%M"),
        })

    df = pd.DataFrame(skill_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(len(skills) * 35 + 38, 220),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM FULL WIDTH — Live Event Timeline
# ─────────────────────────────────────────────────────────────────────────────

def render_event_timeline():
    st.markdown('<div class="section-header">📡 Live Event Timeline</div>', unsafe_allow_html=True)

    bus = _get_bus()
    events = bus.get_history(limit=100)

    if not events:
        st.markdown(
            '<div style="color:#546E7A;font-size:0.8rem;text-align:center;padding:12px 0;">'
            'No events yet — events appear in real-time as missions run.</div>',
            unsafe_allow_html=True,
        )
        return

    import pandas as pd
    rows = []
    for ev in reversed(events):
        ts = datetime.datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S.%f")[:-3]
        msg = ev.data.get("message", ev.event_type)[:90]
        rows.append({
            "Time":      ts,
            "Mission":   ev.mission_id[:12],
            "Agent":     ev.source[:18],
            "Event":     ev.event_type[:22],
            "Detail":    msg,
        })

    df = pd.DataFrame(rows)

    # Row coloring via styling
    def row_style(row):
        et = row["Event"]
        styles = [""] * len(row)
        if "complete" in et or "approved" in et:
            styles = ["color: #56D364"] * len(row)
        elif "error" in et or "failed" in et or "blocked" in et or "offline" in et:
            styles = ["color: #F85149"] * len(row)
        elif "retry" in et or "recovery" in et:
            styles = ["color: #FF9800"] * len(row)
        elif "state_change" in et:
            styles = ["color: #4FC3F7"] * len(row)
        elif "memory" in et or "skill" in et:
            styles = ["color: #FFD54F"] * len(row)
        return styles

    st.dataframe(
        df.style.apply(row_style, axis=1),
        use_container_width=True,
        hide_index=True,
        height=220,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MISSION REPORT viewer
# ─────────────────────────────────────────────────────────────────────────────

def render_report_viewer():
    missions = st.session_state.missions
    complete_missions = [m for m in missions if m.get("state") == "COMPLETE" and m.get("final_report")]

    if not complete_missions:
        return

    st.markdown('<div class="section-header">📄 Mission Reports</div>', unsafe_allow_html=True)

    for m in complete_missions[:3]:
        with st.expander(
            f"✅ {m['id']} — {m['goal'][:60]}...",
            expanded=(m == complete_missions[0]),
        ):
            # Nemotron Vision summary
            vision = m.get("vision_result") or {}
            if vision:
                risk_score = vision.get("overall_risk_score", "N/A")
                scene_sum = vision.get("scene_summary", "")
                immediate = vision.get("immediate_action_required", False)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("🎯 Risk Score", f"{risk_score}/100")
                with c2:
                    st.metric("⚠️ Immediate Action", "YES" if immediate else "No")
                with c3:
                    st.metric("🔍 Anomalies", len(vision.get("anomalies", [])))
                if scene_sum:
                    st.info(f"**Scene:** {scene_sum}")

            # Final report
            report = m.get("final_report", "")
            if report:
                st.markdown(report)

            # Metadata
            st.caption(
                f"Elapsed: {_format_elapsed(m.get('elapsed', 0))} | "
                f"Tokens: {m.get('token_count', 0):,} | "
                f"Input: {m.get('input_type', '').upper()} | "
                f"Retries: {m.get('retry_count', 0)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_rerun():
    """
    Trigger a rerun if a mission is running.
    Called at the end of each render pass.
    """
    if st.session_state.running_mission_id:
        time.sleep(1.5)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Sync events before rendering
    _sync_events()

    # ── Top bar ───────────────────────────────────────────────────────────
    render_topbar()

    # ── Main 3-column layout ──────────────────────────────────────────────
    left_col, center_col, right_col = st.columns([1.0, 3.2, 1.2], gap="medium")

    with left_col:
        render_mission_control()

    with center_col:
        render_agent_graph()

    with right_col:
        render_terminal()

    st.divider()

    # ── Bottom row ────────────────────────────────────────────────────────
    bot_left, bot_right = st.columns([1.5, 1.5], gap="medium")

    with bot_left:
        render_metrics()

    with bot_right:
        render_skill_library()

    st.divider()

    # ── Full-width timeline ────────────────────────────────────────────────
    render_event_timeline()

    st.divider()

    # ── Report viewer ─────────────────────────────────────────────────────
    render_report_viewer()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;color:#546E7A;font-size:0.72rem;padding:16px 0 8px 0;">'
        'FARSIX — Framework for Agentic Reasoning and Supervised Intelligence (X-tended) &nbsp;|&nbsp; '
        'Built by <strong style="color:#8B949E;">Salman Farsi</strong> — '
        'Undergraduate AI Researcher, Brac University &nbsp;|&nbsp; '
        '100% NVIDIA Stack &nbsp;·&nbsp; CPU-Only &nbsp;·&nbsp; Async Multi-Agent'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Auto-refresh when mission running ─────────────────────────────────
    _maybe_rerun()


if __name__ == "__main__":
    main()
