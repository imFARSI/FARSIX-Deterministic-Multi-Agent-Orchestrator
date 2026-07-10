<div align="center">
  <h1>FARSIX: Deterministic VLA Orchestrator & Cognitive Guardrails Engine</h1>
  <p><b>Advanced Agentic Reasoning and Supervised Intelligence Framework</b></p>

  <!-- Autoplaying Demo Video -->
  <video src="docs/assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>
</div>

---

## 📌 The Problem: The Hallucination Bottleneck in Physical AI

In the modern era of Generative AI, wrapping Large Language Models (LLMs) in simple API scripts is sufficient for chatbots, but **catastrophically dangerous for Physical AI** (Robotics, Factory Automation, Autonomous Systems). 

LLMs suffer from non-deterministic logic and hallucinations. If a Vision-Language-Action (VLA) model hallucinates a robotic actuator command or misdiagnoses a factory floor anomaly, the results can lead to critical hardware damage or safety hazards. 

The industry demands a system that bridges the gap between the cognitive reasoning power of modern LLMs (like Llama-3.1-70B) and the absolute strictness of traditional software engineering.

---

## 🚀 The Solution: FARSIX Architecture

**FARSIX** (Framework for Agentic Reasoning and Supervised Intelligence) is a production-grade orchestration pipeline designed specifically to solve the non-deterministic nature of VLA agents. 

It implements a **Dual-Layer Cognitive Architecture**:
1. **The Reasoning Layer (NVIDIA NIM)**: Utilizes high-parameter reasoning models (`Llama-3.1-70B-Instruct`) and specialized vision models (`Nemotron-Nano-VL-8B`) to understand complex spatial environments and generate comprehensive analytical reports.
2. **The Deterministic Guardrails Layer (NVIDIA NeMo)**: A zero-latency, programmatic safety net that intercepts all AI outputs before they are executed.

### Core Breakthrough: The Zero-Latency Input Rail

Instead of relying on a secondary LLM to read and validate the output (which adds latency, cost, and further hallucination risk), FARSIX integrates **NVIDIA NeMo Guardrails** via a custom Colang (`.co`) architecture.

By structuring the Colang rules as an automatic **Input Rail**, FARSIX triggers 100% deterministic Python Regex validations at the exact millisecond the reasoning model generates text. 
- **Latency**: ~0.07 seconds
- **LLM Tokens Consumed**: 0
- **Safety**: Absolute. It physically blocks the mission pipeline and forces a retry checkpoint if safety boundaries are violated.

<div align="center">
  <img src="docs/assets/demo_1.png" width="100%" alt="FARSIX Dynamic SVG Orchestration Graph">
  <br>
  <i>The real-time Orchestration Dashboard tracking the VLA pipeline states.</i>
</div>

---

## 🛠️ Key Features

- **Dynamic State Machine**: A resilient mission engine that handles multi-agent orchestration. If a downstream agent fails, FARSIX intelligently resumes from the last valid checkpoint rather than restarting the entire mission.
- **Visual-Spatial Intelligence**: Integrates NVIDIA's Nano-VL vision model to analyze factory floor diagrams, robotic environments, and sensor feeds.
- **Real-Time SVG Orchestrator**: A custom-built, responsive React/Streamlit dashboard that visually tracks the exact topological state of every agent in the graph.
- **Skill Memory**: Maintains a localized ChromaDB vector store, allowing the orchestrator to dynamically retrieve contextual knowledge based on the current mission profile.

<div align="center">
  <img src="docs/assets/demo_2.png" width="100%" alt="FARSIX System Metrics and Mission Queue">
  <br>
  <i>System telemetry, API tracking, and real-time mission queue execution logs.</i>
</div>

---

## 💼 Use Cases

FARSIX is designed for high-stakes, mission-critical AI applications:
- **Factory Floor Auditing**: Analyzing visual feeds to detect safety hazards or operational anomalies.
- **Robotic QA Systems**: Verifying that a robot's planned actions do not violate geometric or safety constraints before execution.
- **Automated Root-Cause Diagnosis**: Ingesting complex machinery telemetry (CSV/Text) and providing verified, hallucination-free diagnostic reports.

---

## ⚙️ Quick Start Installation

FARSIX is designed to run locally, connecting securely to the NVIDIA NIM cloud API.

### 1. Prerequisites
- Python 3.11+
- An NVIDIA NIM API Key

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/imfarsi/farsix.git
cd farsix

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your NVIDIA API key:
```env
NVIDIA_API_KEY=nvapi-your-key-here
```

### 4. Run the Pipeline
```bash
# Start the FARSIX Orchestration Dashboard
python -m streamlit run app.py
```
Alternatively, on Windows, double-click the `run.bat` file.

---
*Built for the next generation of Physical AI systems.*
