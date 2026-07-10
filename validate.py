"""
FARSIX Import Validation Script
Run: py -3.11 validate.py
Checks all imports and basic connectivity without launching Streamlit.
"""
# encoding: utf-8
import sys
import os

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

print(f"Python: {sys.version}")
print("="*55)

errors = []
warnings = []

def ok(msg):
    print(f"[OK]   {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"[WARN] {msg}")

def fail(msg):
    errors.append(msg)
    print(f"[FAIL] {msg}")


# ── Core stdlib ──────────────────────────────────────────────────────
try:
    import asyncio, sqlite3, threading, json
    ok("stdlib: asyncio, sqlite3, threading, json")
except ImportError as e:
    fail(f"stdlib: {e}")

# ── dotenv ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    ok("python-dotenv: loaded")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if api_key:
        ok(f"NVIDIA_API_KEY: {api_key[:12]}...")
    else:
        warn("NVIDIA_API_KEY not set in .env")
except ImportError as e:
    fail(f"dotenv: {e}")

# ── openai ────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    ok("openai: imported")
except ImportError as e:
    fail(f"openai: {e}")

# ── streamlit ─────────────────────────────────────────────────────────
try:
    import streamlit
    ok(f"streamlit: {streamlit.__version__}")
except ImportError as e:
    fail(f"streamlit: {e}")

# ── plotly ────────────────────────────────────────────────────────────
try:
    import plotly
    ok(f"plotly: {plotly.__version__}")
except ImportError as e:
    fail(f"plotly: {e}")

# ── pandas ────────────────────────────────────────────────────────────
try:
    import pandas as pd
    ok(f"pandas: {pd.__version__}")
except ImportError as e:
    fail(f"pandas: {e}")

# ── psutil ────────────────────────────────────────────────────────────
try:
    import psutil
    cpu = psutil.cpu_percent(0.1)
    ok(f"psutil: CPU={cpu:.0f}%")
except ImportError as e:
    fail(f"psutil: {e}")

# ── pdfplumber ────────────────────────────────────────────────────────
try:
    import pdfplumber
    ok("pdfplumber: imported")
except ImportError as e:
    fail(f"pdfplumber: {e}")

# ── pillow ────────────────────────────────────────────────────────────
try:
    from PIL import Image
    ok("pillow: imported")
except ImportError as e:
    fail(f"pillow: {e}")

# ── networkx ─────────────────────────────────────────────────────────
try:
    import networkx as nx
    ok(f"networkx: {nx.__version__}")
except ImportError as e:
    fail(f"networkx: {e}")

# ── pyvis ─────────────────────────────────────────────────────────────
try:
    from pyvis.network import Network
    ok("pyvis: imported")
except ImportError as e:
    warn(f"pyvis: {e} (graph will use SVG fallback)")

# ── chromadb ─────────────────────────────────────────────────────────
try:
    import chromadb
    ok(f"chromadb: {chromadb.__version__}")
except ImportError as e:
    warn(f"chromadb: {e} (memory disabled)")

# ── nemoguardrails ────────────────────────────────────────────────────
try:
    import nemoguardrails
    ok("nemoguardrails: imported")
except ImportError as e:
    warn(f"nemoguardrails: {e} (Python rule-checking only)")

# ── FARSIX backend ────────────────────────────────────────────────────
print("\n-- FARSIX Backend --")
try:
    from backend.event_bus import get_event_bus
    bus = get_event_bus()
    ok("EventBus: initialised")
except Exception as e:
    fail(f"EventBus: {e}")

try:
    from backend.nim_router import NIMRouter
    router = NIMRouter()
    d = router.route("vision")
    ok(f"NIMRouter: vision -> {d.model_id}")
except Exception as e:
    fail(f"NIMRouter: {e}")

try:
    from backend.skill_library import get_skill_library
    lib = get_skill_library()
    ok(f"SkillLibrary: {lib.skill_count()} skills in DB")
except Exception as e:
    fail(f"SkillLibrary: {e}")

try:
    from backend.recovery import get_recovery_engine
    rec = get_recovery_engine()
    ok("RecoveryEngine: initialised")
except Exception as e:
    fail(f"RecoveryEngine: {e}")

try:
    from utils.input_parser import parse_text
    text, meta = parse_text("Factory floor anomaly: motor overheating detected")
    ok(f"InputParser: text parsed ({meta})")
except Exception as e:
    fail(f"InputParser: {e}")

try:
    from utils.metrics import get_metrics
    m = get_metrics()
    m.add_tokens(100)
    ok(f"MetricsCollector: tokens={m.total_tokens()}, cpu={m.cpu_percent():.0f}%")
except Exception as e:
    fail(f"Metrics: {e}")

try:
    from utils.graph_viz import build_agent_graph
    html = build_agent_graph({})
    ok(f"GraphViz: graph built ({len(html)} chars)")
except Exception as e:
    warn(f"GraphViz: {e}")

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"[FAIL] {len(errors)} critical error(s):")
    for e in errors:
        print(f"  - {e}")
    print("\nInstall missing packages:")
    print("  py -3.11 -m pip install -r requirements.txt")
else:
    print("[OK] All critical imports passed!")

if warnings:
    print(f"\n[WARN] {len(warnings)} warning(s) (non-fatal):")
    for w in warnings:
        print(f"  - {w}")

print("\nTo launch FARSIX:")
print("  py -3.11 -m streamlit run app.py")
print("  or double-click run.bat")
