"""
FARSIX Animated Agent Orchestration Graph — Pure SVG + CSS

Builds a highly animated, tech-vibe SVG visualization showing:
  - Hexagonal agent nodes with glowing neon borders
  - Animated data-flow particles traveling along connection paths
  - Pulsing radar effects on RUNNING agents
  - Sequential pipeline: INPUT → VISION → REASONING → GUARDRAILS → REPORT
  - Color-coded by state: IDLE=green, RUNNING=blue pulse, COMPLETE=teal, OFFLINE=red

Returns an HTML string rendered via st.components.v1.html().
"""

from typing import Dict, List, Optional, Tuple


# State → color mapping
STATE_COLORS = {
    "IDLE":    "#4CAF50",
    "RUNNING": "#2196F3",
    "COMPLETE":"#00BCD4",
    "OFFLINE": "#F44336",
    "FAILED":  "#FF5722",
    "UNKNOWN": "#9E9E9E",
}

# Node definitions: id → (label, sublabel, color, cx, cy)
NODES = {
    "nim_router":       ("NIM ROUTER",       "Task Dispatcher",                  "#FFD54F", 500, 60),
    "vision_agent":     ("NEMOTRON VISION",   "nemotron-nano-vl-8b",              "#4FC3F7", 180, 200),
    "nemotron_agent":   ("NEMOTRON 70B",      "Deep Reasoning",                   "#CE93D8", 820, 200),
    "guardrails_agent": ("GUARDRAILS",        "NeMo Colang CPU",                  "#EF9A9A", 500, 340),
    "llama_agent":      ("LLAMA 8B",          "Report Generator",                 "#A5D6A7", 500, 470),
}

# Pipeline edges: (source_id, target_id)
EDGES = [
    ("nim_router",       "vision_agent"),
    ("nim_router",       "nemotron_agent"),
    ("vision_agent",     "nemotron_agent"),
    ("vision_agent",     "guardrails_agent"),
    ("nemotron_agent",   "guardrails_agent"),
    ("guardrails_agent", "llama_agent"),
]


def _hex_path(cx: int, cy: int, r: int = 48) -> str:
    """Generate SVG path for a hexagon centered at (cx, cy)."""
    import math
    points = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append(f"{px:.1f},{py:.1f}")
    return "M" + "L".join(points) + "Z"


def _edge_path(x1: int, y1: int, x2: int, y2: int) -> str:
    """Generate a curved SVG path between two points."""
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    # Add slight curve
    dx = x2 - x1
    dy = y2 - y1
    cx = mx - dy * 0.15
    cy = my + dx * 0.15
    return f"M{x1},{y1} Q{cx:.0f},{cy:.0f} {x2},{y2}"


def build_agent_graph(
    agent_states: Optional[Dict[str, str]] = None,
    active_edges: Optional[list] = None,
    height: int = 480,
) -> str:
    """
    Build an animated SVG + CSS agent orchestration graph.

    Args:
        agent_states:  Dict mapping agent_name → state string
        active_edges:  List of (source, target) tuples to highlight
        height:        Graph container height in pixels

    Returns:
        HTML string for st.components.v1.html().
    """
    states = agent_states or {}
    active_set = set(map(tuple, active_edges)) if active_edges else set()

    w = 1000

    # ── Build edge SVG elements ──────────────────────────────────────────
    edges_svg = ""
    particles_svg = ""
    for idx, (src_id, tgt_id) in enumerate(EDGES):
        src = NODES[src_id]
        tgt = NODES[tgt_id]
        x1, y1 = src[3], src[4]
        x2, y2 = tgt[3], tgt[4]

        is_active = (src_id, tgt_id) in active_set
        path_d = _edge_path(x1, y1, x2, y2)
        path_id = f"edge-{idx}"

        if is_active:
            # Glowing active edge
            edges_svg += f'''
            <path d="{path_d}" fill="none" stroke="#FFD54F" stroke-width="3"
                  filter="url(#glow-yellow)" opacity="0.9" class="edge-active"/>
            '''
            # Animated particle along the path
            particles_svg += f'''
            <circle r="5" fill="#FFD54F" filter="url(#glow-yellow)" class="particle">
              <animateMotion dur="1.2s" repeatCount="indefinite" path="{path_d}"/>
            </circle>
            <circle r="3" fill="#fff" opacity="0.9">
              <animateMotion dur="1.2s" repeatCount="indefinite" path="{path_d}"/>
            </circle>
            '''
        else:
            edges_svg += f'''
            <path d="{path_d}" fill="none" stroke="#1E3A5F" stroke-width="1.5"
                  stroke-dasharray="6,4" opacity="0.5"/>
            '''

    # ── Build node SVG elements ──────────────────────────────────────────
    nodes_svg = ""
    for node_id, (label, sublabel, default_color, cx, cy) in NODES.items():
        state = states.get(node_id, "IDLE").upper()
        color = STATE_COLORS.get(state, STATE_COLORS["IDLE"])
        hex_d = _hex_path(cx, cy, r=48)

        is_running = state == "RUNNING"
        is_complete = state == "COMPLETE"

        # Radar pulse for running agents
        radar = ""
        if is_running:
            radar = f'''
            <circle cx="{cx}" cy="{cy}" r="48" fill="none" stroke="{color}"
                    stroke-width="2" opacity="0" class="radar-ping"/>
            <circle cx="{cx}" cy="{cy}" r="48" fill="none" stroke="{color}"
                    stroke-width="1.5" opacity="0" class="radar-ping" style="animation-delay:0.6s;"/>
            '''

        # Glow filter ref
        glow_ref = ""
        if is_running:
            glow_ref = 'filter="url(#glow-blue)"'
        elif is_complete:
            glow_ref = 'filter="url(#glow-green)"'

        # State indicator
        state_emoji = {"IDLE": "●", "RUNNING": "◉", "COMPLETE": "✓", "OFFLINE": "✗", "FAILED": "✗"}.get(state, "●")
        state_color_indicator = color

        # Build node
        nodes_svg += f'''
        {radar}
        <g class="agent-node {'node-running' if is_running else ''}" data-state="{state}">
          <!-- Hex background -->
          <path d="{hex_d}" fill="#0D1117" stroke="{color}" stroke-width="{'3' if is_running else '2'}"
                {glow_ref} opacity="0.95"/>
          <!-- Inner hex accent -->
          <path d="{_hex_path(cx, cy, r=40)}" fill="none" stroke="{color}" stroke-width="0.5" opacity="0.3"/>

          <!-- Label -->
          <text x="{cx}" y="{cy - 10}" text-anchor="middle" fill="{color}"
                font-family="'Inter', 'Segoe UI', sans-serif" font-size="10" font-weight="700"
                letter-spacing="0.5">{label}</text>
          <!-- Sublabel -->
          <text x="{cx}" y="{cy + 6}" text-anchor="middle" fill="#8B949E"
                font-family="'JetBrains Mono', monospace" font-size="8">{sublabel}</text>
          <!-- State -->
          <text x="{cx}" y="{cy + 22}" text-anchor="middle" fill="{state_color_indicator}"
                font-family="'JetBrains Mono', monospace" font-size="9" font-weight="600">
            {state_emoji} {state}
          </text>
        </g>
        '''

    # ── Assemble full HTML ───────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body {{
  margin: 0; padding: 0;
  background: #0D1117;
  overflow: hidden;
}}

svg {{
  display: block;
  margin: 0 auto;
}}

/* ── Radar ping animation ─────────────────────── */
.radar-ping {{
  animation: radar-expand 1.8s ease-out infinite;
}}

@keyframes radar-expand {{
  0%   {{ r: 48; opacity: 0.6; stroke-width: 2; }}
  100% {{ r: 80; opacity: 0;   stroke-width: 0.5; }}
}}

/* ── Running node subtle pulse ────────────────── */
.node-running {{
  animation: node-pulse 2s ease-in-out infinite;
}}

@keyframes node-pulse {{
  0%, 100% {{ opacity: 1; }}
  50%      {{ opacity: 0.85; }}
}}

/* ── Particle glow ────────────────────────────── */
.particle {{
  animation: particle-glow 0.8s ease-in-out infinite alternate;
}}

@keyframes particle-glow {{
  0%   {{ r: 4; opacity: 0.7; }}
  100% {{ r: 6; opacity: 1; }}
}}

/* ── Active edge shimmer ──────────────────────── */
.edge-active {{
  animation: edge-shimmer 1.5s ease-in-out infinite alternate;
}}

@keyframes edge-shimmer {{
  0%   {{ stroke-width: 2; opacity: 0.6; }}
  100% {{ stroke-width: 4; opacity: 1; }}
}}

/* ── Grid background animation ────────────────── */
.grid-line {{
  animation: grid-fade 4s ease-in-out infinite alternate;
}}

@keyframes grid-fade {{
  0%   {{ opacity: 0.03; }}
  100% {{ opacity: 0.08; }}
}}

/* ── Corner decorations pulse ─────────────────── */
.corner-deco {{
  animation: corner-pulse 3s ease-in-out infinite alternate;
}}

@keyframes corner-pulse {{
  0%   {{ opacity: 0.15; }}
  100% {{ opacity: 0.35; }}
}}
</style></head>
<body>
<svg width="100%" height="100%" viewBox="0 0 {w} {height}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
  <defs>
    <!-- Glow filters -->
    <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feFlood flood-color="#4FC3F7" flood-opacity="0.5" result="color"/>
      <feComposite in="color" in2="blur" operator="in" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feFlood flood-color="#00BCD4" flood-opacity="0.4" result="color"/>
      <feComposite in="color" in2="blur" operator="in" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="glow-yellow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feFlood flood-color="#FFD54F" flood-opacity="0.5" result="color"/>
      <feComposite in="color" in2="blur" operator="in" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feFlood flood-color="#F44336" flood-opacity="0.5" result="color"/>
      <feComposite in="color" in2="blur" operator="in" result="glow"/>
      <feMerge>
        <feMergeNode in="glow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="{w}" height="{height}" fill="#0D1117" rx="12"/>

  <!-- Subtle grid pattern -->
  <g class="grid-line">
    {"".join(f'<line x1="0" y1="{y}" x2="{w}" y2="{y}" stroke="#1E3A5F" stroke-width="0.5"/>' for y in range(0, height, 40))}
    {"".join(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="#1E3A5F" stroke-width="0.5"/>' for x in range(0, w, 40))}
  </g>

  <!-- Corner tech decorations -->
  <g class="corner-deco">
    <path d="M10,30 L10,10 L30,10" fill="none" stroke="#4FC3F7" stroke-width="1.5"/>
    <path d="M{w-10},{height-30} L{w-10},{height-10} L{w-30},{height-10}" fill="none" stroke="#CE93D8" stroke-width="1.5"/>
    <path d="M{w-10},30 L{w-10},10 L{w-30},10" fill="none" stroke="#4FC3F7" stroke-width="1.5"/>
    <path d="M10,{height-30} L10,{height-10} L30,{height-10}" fill="none" stroke="#CE93D8" stroke-width="1.5"/>
  </g>

  <!-- Title -->
  <text x="{w//2}" y="28" text-anchor="middle" fill="#8B949E"
        font-family="'JetBrains Mono', monospace" font-size="10" letter-spacing="3" opacity="0.6">
    AGENT ORCHESTRATION PIPELINE
  </text>

  <!-- Edges -->
  {edges_svg}

  <!-- Nodes -->
  {nodes_svg}

  <!-- Particles (on top) -->
  {particles_svg}

</svg>
</body></html>'''

    return html


def get_state_color(state: str) -> str:
    return STATE_COLORS.get(state.upper(), STATE_COLORS["UNKNOWN"])


def get_state_emoji(state: str) -> str:
    emojis = {
        "IDLE": "🟢", "RUNNING": "🔵", "COMPLETE": "✅",
        "OFFLINE": "🔴", "FAILED": "⚠️", "UNKNOWN": "⚪",
    }
    return emojis.get(state.upper(), "⚪")
