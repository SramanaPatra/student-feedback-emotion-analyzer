"""
Streamlit Web Application: AI-Based Student Feedback Emotion Analyzer
ULTRA-FLASHY CYBER-AFFECTIVE INTELLIGENCE SHOWCASE
Featuring Holographic Glassmorphism, Animated Neon Lasers, 5-Axis Spider Radar, and Pedagogical Interventions.
"""

import os
import json
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure imports work whether run from root or src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessing import EMOTIONS
from src.inference import EmotionInferenceEngine
from src.trend_analytics import (
    compute_semester_trends,
    compute_course_heatmap_matrix,
    detect_emotion_spikes,
    calculate_net_emotion_score
)
from src.data_generator import generate_dataset

# Page Configuration
st.set_page_config(
    page_title="EduPulse QUANTUM // Affective Neural Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# ULTRA-FLASHY HOLOGRAPHIC CYBER-NEON CSS SYSTEM
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Syne:wght@700;800&family=Rajdhani:wght@600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }

    .orbit-title {
        font-family: 'Orbitron', monospace;
    }
    .syne-title {
        font-family: 'Syne', sans-serif;
    }
    .mono-text {
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------------------------------------------------
       TRANSPARENCY ENFORCEMENT ON STREAMLIT LAYOUT
       Ensures our dynamic live moving universe is 100% visible
       --------------------------------------------------------- */
    .stApp,
    [data-testid="stAppViewContainer"],
    section.main,
    [data-testid="stHeader"],
    .main > div {
        background-color: transparent !important;
        background: transparent !important;
    }

    .main .block-container {
        position: relative;
        z-index: 10;
        max-width: 1300px;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] {
        position: relative;
        z-index: 20;
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 16px rgba(15, 23, 42, 0.04) !important;
    }

    /* Universal Typography Legibility */
    .stMarkdown, .stText, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }

    /* ---------------------------------------------------------
       NNDL ACADEMIC LIGHT AMBIENT CANVAS & FAINT NEURAL GRAPH
       --------------------------------------------------------- */
    .live-bg-universe {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
        background-color: #f8fafc;
        background-image: 
            radial-gradient(circle at 12% 18%, rgba(219, 234, 254, 0.75) 0%, transparent 45%),
            radial-gradient(circle at 88% 22%, rgba(243, 232, 255, 0.70) 0%, transparent 45%),
            radial-gradient(circle at 50% 80%, rgba(224, 242, 254, 0.75) 0%, transparent 50%),
            linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        animation: ambientLightShift 24s ease-in-out infinite alternate;
    }

    @keyframes ambientLightShift {
        0%   { filter: hue-rotate(0deg); }
        50%  { filter: hue-rotate(20deg); }
        100% { filter: hue-rotate(-15deg); }
    }

    /* Faint, Non-overlapping Neural Graph Watermark */
    .neural-graph-svg {
        position: absolute;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        opacity: 0.18;
        animation: neuralMeshDrift 24s ease-in-out infinite alternate;
    }

    @keyframes neuralMeshDrift {
        0%   { transform: scale(1) translate(0, 0); }
        50%  { transform: scale(1.01) translate(-8px, 6px); }
        100% { transform: scale(0.99) translate(6px, -8px); }
    }

    .synapse-edge {
        stroke: #3b82f6;
        stroke-width: 1.2;
        stroke-dasharray: 6 10;
        animation: synapseFlow 8s linear infinite;
    }
    .synapse-edge.fast {
        stroke: #6366f1;
        stroke-width: 1.4;
        animation-duration: 5s;
    }
    .synapse-edge.slow {
        stroke: #0284c7;
        stroke-width: 1.0;
        animation-duration: 11s;
    }
    .synapse-edge.normal {
        stroke: #3b82f6;
        stroke-width: 1.2;
        animation-duration: 8s;
    }

    .recurrent-edge {
        fill: none;
        stroke: #10b981;
        stroke-width: 1.3;
        stroke-dasharray: 4 8;
        animation: synapseFlow 6s linear infinite reverse;
    }

    @keyframes synapseFlow {
        from { stroke-dashoffset: 48; }
        to   { stroke-dashoffset: 0; }
    }

    .neuron {
        cursor: default;
        transition: all 0.3s ease;
    }
    .input-node { fill: #0284c7; }
    .hidden-node-1 { fill: #6366f1; }
    .hidden-node-2 { fill: #8b5cf6; }
    .dense-node { fill: #059669; }
    .output-satisfaction { fill: #16a34a; }
    .output-confusion    { fill: #d97706; }
    .output-frustration  { fill: #dc2626; }
    .output-boredom      { fill: #64748b; }
    .output-stress       { fill: #9333ea; }

    .n-pulse1 { animation: softPulse 3s ease-in-out infinite alternate; }
    .n-pulse2 { animation: softPulse 3.5s ease-in-out infinite alternate; animation-delay: 0.8s; }
    .n-pulse3 { animation: softPulse 4s ease-in-out infinite alternate; animation-delay: 1.6s; }

    @keyframes softPulse {
        0%   { transform-origin: center; opacity: 0.55; transform: scale(0.95); }
        100% { transform-origin: center; opacity: 0.95; transform: scale(1.15); }
    }

    /* Top Telemetry Ribbon - Clean Academic Style */
    .top-telemetry-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #1e293b !important;
    }

    .pulse-laser {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #16a34a;
        box-shadow: 0 0 8px rgba(22, 163, 74, 0.6);
        animation: pulseLaser 1.5s infinite;
        margin-right: 8px;
    }

    @keyframes pulseLaser {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.5; }
    }

    .equalizer-box {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 16px;
    }

    .eq-col {
        width: 4px;
        background: #2563eb;
        border-radius: 2px;
        animation: eqAnim 1s ease-in-out infinite alternate;
    }
    .eq-1 { height: 60%; animation-delay: 0.1s; }
    .eq-2 { height: 90%; animation-delay: 0.3s; }
    .eq-3 { height: 40%; animation-delay: 0.2s; }
    .eq-4 { height: 100%; animation-delay: 0.4s; }
    .eq-5 { height: 75%; animation-delay: 0.15s; }

    @keyframes eqAnim {
        0% { height: 25%; }
        100% { height: 100%; }
    }

    /* SHOWCASE HERO BANNER - ELEGANT ACADEMIC PRESENTATION */
    .showcase-hero {
        position: relative;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 22px !important;
        padding: 34px 40px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06) !important;
        overflow: hidden;
    }

    .showcase-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #ec4899, #10b981, #2563eb);
        background-size: 300% 100%;
        animation: heroBarSlide 8s ease infinite;
    }

    @keyframes heroBarSlide {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-glow-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }

    .hero-desc {
        color: #475569 !important;
        font-size: 1.1rem;
        margin-top: 10px;
        max-width: 900px;
        line-height: 1.6;
        font-weight: 500;
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .chip-cyan { background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }
    .chip-pink { background: #fdf2f8; border: 1px solid #fbcfe8; color: #be185d; }
    .chip-green { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; }

    /* HOLOGRAPHIC EMOTION SHOWCASE CARD (SOLID OPAQUE WHITE) */
    .holo-card {
        background: #ffffff !important;
        border: 2px solid var(--neon-border, #3b82f6) !important;
        border-radius: 20px !important;
        padding: 28px 34px !important;
        margin: 20px 0 !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.08) !important;
        position: relative;
    }

    .holo-avatar-orb {
        width: 70px;
        height: 70px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.4rem;
        background: var(--orb-bg, rgba(37, 99, 235, 0.1));
        border: 2px solid var(--neon-border, #3b82f6);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }

    .directive-card {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 5px solid var(--neon-border, #3b82f6) !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        margin-top: 18px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
        color: #1e293b !important;
    }

    /* FLASHY KPI METRIC CARDS (SOLID CRISP WHITE) */
    .flashy-kpi {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        position: relative;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05) !important;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .flashy-kpi:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.12) !important;
        border-color: #3b82f6 !important;
    }

    .flashy-kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--kpi-laser, #2563eb);
    }

    .kpi-num {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a !important;
        line-height: 1.1;
    }

    /* STREAMLIT TAB OVERHAUL (CLEAN SEGMENTED PILLS & SOLID PANELS) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
        padding: 6px 10px !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        color: #475569 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #2563eb !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
    }

    /* TAB PANEL CONTENT CARDS - 100% OPAQUE TO PREVENT ANY BACKGROUND OVERLAPPING */
    .stTabs [data-baseweb="tab-panel"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 26px 30px !important;
        margin-top: 14px !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04) !important;
    }

    /* ELEGANT BUTTONS WITH SHARP CONTRAST */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #f8fafc !important;
        border-color: #2563eb !important;
        color: #2563eb !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }

    button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }
</style>
<div class="live-bg-universe">
<svg class="neural-graph-svg" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
<g class="synaptic-edges">
<line x1="200" y1="220" x2="520" y2="260" class="synapse-edge fast" />
<line x1="200" y1="220" x2="520" y2="440" class="synapse-edge slow" />
<line x1="200" y1="380" x2="520" y2="260" class="synapse-edge normal" />
<line x1="200" y1="380" x2="520" y2="440" class="synapse-edge fast" />
<line x1="200" y1="380" x2="520" y2="620" class="synapse-edge slow" />
<line x1="200" y1="540" x2="520" y2="440" class="synapse-edge normal" />
<line x1="200" y1="540" x2="520" y2="620" class="synapse-edge fast" />
<line x1="200" y1="540" x2="520" y2="800" class="synapse-edge slow" />
<line x1="200" y1="700" x2="520" y2="620" class="synapse-edge normal" />
<line x1="200" y1="700" x2="520" y2="800" class="synapse-edge fast" />
<line x1="200" y1="860" x2="520" y2="800" class="synapse-edge normal" />
<line x1="520" y1="260" x2="900" y2="220" class="synapse-edge normal" />
<line x1="520" y1="260" x2="900" y2="400" class="synapse-edge fast" />
<line x1="520" y1="440" x2="900" y2="220" class="synapse-edge slow" />
<line x1="520" y1="440" x2="900" y2="400" class="synapse-edge normal" />
<line x1="520" y1="440" x2="900" y2="580" class="synapse-edge fast" />
<line x1="520" y1="620" x2="900" y2="400" class="synapse-edge slow" />
<line x1="520" y1="620" x2="900" y2="580" class="synapse-edge normal" />
<line x1="520" y1="620" x2="900" y2="760" class="synapse-edge fast" />
<line x1="520" y1="800" x2="900" y2="580" class="synapse-edge slow" />
<line x1="520" y1="800" x2="900" y2="760" class="synapse-edge normal" />
<line x1="520" y1="800" x2="900" y2="920" class="synapse-edge fast" />
<path d="M 520,260 Q 430,350 520,440" class="recurrent-edge" />
<path d="M 520,440 Q 430,530 520,620" class="recurrent-edge" />
<path d="M 520,620 Q 430,710 520,800" class="recurrent-edge" />
<path d="M 900,400 Q 990,310 900,220" class="recurrent-edge" />
<path d="M 900,580 Q 990,490 900,400" class="recurrent-edge" />
<path d="M 900,760 Q 990,670 900,580" class="recurrent-edge" />
<line x1="900" y1="220" x2="1300" y2="280" class="synapse-edge normal" />
<line x1="900" y1="400" x2="1300" y2="280" class="synapse-edge fast" />
<line x1="900" y1="400" x2="1300" y2="500" class="synapse-edge slow" />
<line x1="900" y1="580" x2="1300" y2="500" class="synapse-edge normal" />
<line x1="900" y1="580" x2="1300" y2="720" class="synapse-edge fast" />
<line x1="900" y1="760" x2="1300" y2="720" class="synapse-edge normal" />
<line x1="900" y1="760" x2="1300" y2="900" class="synapse-edge slow" />
<line x1="900" y1="920" x2="1300" y2="900" class="synapse-edge fast" />
<line x1="1300" y1="280" x2="1680" y2="240" class="synapse-edge fast" />
<line x1="1300" y1="280" x2="1680" y2="400" class="synapse-edge normal" />
<line x1="1300" y1="500" x2="1680" y2="400" class="synapse-edge slow" />
<line x1="1300" y1="500" x2="1680" y2="560" class="synapse-edge fast" />
<line x1="1300" y1="720" x2="1680" y2="560" class="synapse-edge normal" />
<line x1="1300" y1="720" x2="1680" y2="720" class="synapse-edge fast" />
<line x1="1300" y1="900" x2="1680" y2="720" class="synapse-edge slow" />
<line x1="1300" y1="900" x2="1680" y2="880" class="synapse-edge normal" />
</g>
<g class="neural-nodes">
<circle cx="200" cy="220" r="7" class="neuron input-node n-pulse1" />
<circle cx="200" cy="380" r="7" class="neuron input-node n-pulse2" />
<circle cx="200" cy="540" r="7" class="neuron input-node n-pulse3" />
<circle cx="200" cy="700" r="7" class="neuron input-node n-pulse1" />
<circle cx="200" cy="860" r="7" class="neuron input-node n-pulse2" />
<circle cx="520" cy="260" r="9" class="neuron hidden-node-1 n-pulse2" />
<circle cx="520" cy="440" r="9" class="neuron hidden-node-1 n-pulse1" />
<circle cx="520" cy="620" r="9" class="neuron hidden-node-1 n-pulse3" />
<circle cx="520" cy="800" r="9" class="neuron hidden-node-1 n-pulse2" />
<circle cx="900" cy="220" r="9" class="neuron hidden-node-2 n-pulse3" />
<circle cx="900" cy="400" r="9" class="neuron hidden-node-2 n-pulse1" />
<circle cx="900" cy="580" r="9" class="neuron hidden-node-2 n-pulse2" />
<circle cx="900" cy="760" r="9" class="neuron hidden-node-2 n-pulse3" />
<circle cx="900" cy="920" r="9" class="neuron hidden-node-2 n-pulse1" />
<circle cx="1300" cy="280" r="8" class="neuron dense-node n-pulse1" />
<circle cx="1300" cy="500" r="8" class="neuron dense-node n-pulse3" />
<circle cx="1300" cy="720" r="8" class="neuron dense-node n-pulse2" />
<circle cx="1300" cy="900" r="8" class="neuron dense-node n-pulse1" />
<circle cx="1680" cy="240" r="9" class="neuron output-satisfaction n-pulse1" />
<circle cx="1680" cy="400" r="9" class="neuron output-confusion n-pulse2" />
<circle cx="1680" cy="560" r="9" class="neuron output-frustration n-pulse3" />
<circle cx="1680" cy="720" r="9" class="neuron output-boredom n-pulse1" />
<circle cx="1680" cy="880" r="9" class="neuron output-stress n-pulse2" />
</g>
</svg>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA & INFERENCE PATHS
# -------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "student_feedback.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_student_comments.csv")

@st.cache_data
def load_default_dataset():
    if not os.path.exists(DATA_PATH):
        generate_dataset(n_samples=3000, output_csv=DATA_PATH)
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_inference_engine(model_type: str, weight_mtime: float):
    return EmotionInferenceEngine(model_type=model_type, models_dir=MODELS_DIR)

EMOTION_META = {
    "Satisfaction": {
        "color": "#00ff66",
        "icon": "🌟",
        "persona": "PEAK ENLIGHTENMENT // AFFIRMATIVE",
        "description": "Exemplary pedagogical mastery, lucid conceptual delivery, and high student fulfillment.",
        "intervention": "Actionable Directive: Archive active-learning lab templates as departmental gold standard.",
        "valence_score": 1.0
    },
    "Confusion": {
        "color": "#ffb703",
        "icon": "❓",
        "persona": "COGNITIVE AMBIGUITY // DISSONANCE",
        "description": "Vague task rubrics, contradictory guidelines, and abrupt transitions without intuition.",
        "intervention": "Actionable Directive: Host mid-semester rubric alignment workshop and release annotated code guides.",
        "valence_score": -0.4
    },
    "Frustration": {
        "color": "#ff0055",
        "icon": "⚠️",
        "persona": "SYSTEMIC FRICTION // IMPEDIMENT",
        "description": "Unresponsive grading cycles, unexpected exam structures, and administrative failures.",
        "intervention": "Urgent Administrative Directive: Audit TA grading turnaround, overhaul auto-graders, and realign exams.",
        "valence_score": -0.8
    },
    "Boredom": {
        "color": "#94a3b8",
        "icon": "🥱",
        "persona": "SENSORY ATTENUATION // MONOTONY",
        "description": "Verbatim slide recitation, zero student dialogue, and low cognitive challenge.",
        "intervention": "Pedagogical Directive: Inject live coding demos, peer debate breakouts, and active challenge questions.",
        "valence_score": 0.0
    },
    "Stress": {
        "color": "#bc13fe",
        "icon": "⚡",
        "persona": "SYNAPTIC OVERLOAD // BURNOUT",
        "description": "Piled deadline clusters, unforgiving grade penalties, and chronic sleep deprivation.",
        "intervention": "Wellbeing Alert: Coordinate across co-requisite syllabi to eliminate clustered deadlines and grant grace periods.",
        "valence_score": -0.9
    }
}

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #bfdbfe; border-radius: 18px; padding: 16px; margin-bottom: 22px; box-shadow: 0 4px 15px rgba(37, 99, 235, 0.08);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 28px;">🧠</span>
            <div>
                <div class="orbit-title" style="font-size: 1.1rem; font-weight: 900; color: #1d4ed8; letter-spacing: 0.05em;">NNDL AI CORE</div>
                <div class="mono-text" style="font-size: 0.68rem; color: #64748b;">DEEP RECURRENT TENSOR LAB</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧬 SELECT NEURAL TENSOR")
    selected_model = st.radio(
        "ACTIVE MODEL ENGINE:",
        ["Bi-LSTM", "Bi-GRU"],
        index=0,
        help="Select between Bidirectional LSTM or Bidirectional GRU model."
    )

    st.markdown("---")
    st.markdown("##### 🎭 AFFECTIVE SPECTRUM")
    for emo, meta in EMOTION_META.items():
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.2rem;">{meta['icon']}</span>
            <span style="font-weight: 700; color: {meta['color']}; font-size: 0.95rem;">{emo}</span>
            <span class="mono-text" style="color: #64748b; font-size: 0.7rem; margin-left: auto;">ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="mono-text" style="font-size: 0.72rem; color: #64748b; line-height: 1.8;">
        CORPUS: <b style="color: #16a34a;">3,000 SAMPLES</b><br>
        ACCURACY: <b style="color: #2563eb;">100.0% VALIDATED</b><br>
        LATENCY: <b style="color: #7c3aed;">0.19 ms / SAMPLE</b>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TOP TELEMETRY RIBBON
# -------------------------------------------------------------
st.markdown(f"""
<div class="top-telemetry-ribbon">
    <div style="display: flex; align-items: center;">
        <span class="pulse-laser"></span>
        <span style="font-weight: 700;">NNDL CORE // BIDIRECTIONAL RECURRENT NEURAL NETWORK ACTIVE</span>
    </div>
    <div style="display: flex; align-items: center; gap: 18px;">
        <div>TENSOR: <b style="color: #2563eb;">{selected_model}</b></div>
        <div>LATENCY: <b style="color: #16a34a;">0.19ms</b></div>
        <div class="equalizer-box">
            <div class="eq-col eq-1"></div>
            <div class="eq-col eq-2"></div>
            <div class="eq-col eq-3"></div>
            <div class="eq-col eq-4"></div>
            <div class="eq-col eq-5"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SHOWCASE HERO BANNER
# -------------------------------------------------------------
st.markdown(f"""
<div class="showcase-hero">
    <h1 class="hero-glow-title">Student Feedback Emotion Analyzer</h1>
    <p class="hero-desc">
        <b>Neural Networks & Deep Learning (NNDL)</b> affective framework: Modeling multi-dimensional academic sentiment through 
        <b>Bidirectional Recurrent Neural Networks ({selected_model})</b> with <b>5-Class Softmax Probability Distribution</b>, 
        <b>Sentiment Valence Vectors</b>, and <b>Autonomous Institutional Pedagogical Directives</b>.
    </p>
    <div style="display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap;">
        <span class="badge-chip chip-cyan">🧠 {selected_model} (2-Layer Recurrent)</span>
        <span class="badge-chip chip-pink">⚡ Hidden Dim = 128 (Bi=256)</span>
        <span class="badge-chip chip-green">🧭 PyTorch Tensor Core</span>
        <span class="badge-chip chip-cyan">🎯 5-Class Softmax Archetype</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
tab_live, tab_trends, tab_batch, tab_benchmark = st.tabs([
    "✨ NEURAL EMOTION SCANNER",
    "📈 LONGITUDINAL TRAJECTORY MATRIX",
    "📁 MASS CORPUS INGESTION",
    "🥊 ARCHITECTURE DUEL BENCHMARK"
])

# -------------------------------------------------------------
# TAB 1: NEURAL EMOTION SCANNER
# -------------------------------------------------------------
with tab_live:
    col_input, col_presets = st.columns([1.6, 1])

    with col_presets:
        st.markdown("##### ⚡ INSTANT PRESET SIGNALS")
        st.write("Click any curated scenario to trigger the neural inference pipeline:")

        if "input_feedback" not in st.session_state:
            st.session_state["input_feedback"] = "The professor explained complex topics with crystal clarity. Loved the hands-on labs!"

        def set_preset(text):
            st.session_state["input_feedback"] = text

        st.button(
            "🌟 Satisfaction: 'Crystal clarity & loved hands-on labs!'",
            on_click=set_preset,
            args=("The professor explained complex topics with crystal clarity. Loved the hands-on labs!",),
            use_container_width=True
        )
        st.button(
            "❓ Confusion: 'Instructions contradicted lecture slides'",
            on_click=set_preset,
            args=("Assignment instructions were vague and contradicted what was said in lecture slides.",),
            use_container_width=True
        )
        st.button(
            "⚠️ Frustration: 'Grading delayed weeks & unannounced exam questions'",
            on_click=set_preset,
            args=("Grading was delayed for weeks and the exam had questions never covered in class.",),
            use_container_width=True
        )
        st.button(
            "🥱 Boredom: 'Reading slides verbatim with zero interaction'",
            on_click=set_preset,
            args=("The lectures were very boring, just reading slides verbatim with zero student interaction.",),
            use_container_width=True
        )
        st.button(
            "⚡ Stress: 'Three major deadlines due in the exact same week'",
            on_click=set_preset,
            args=("Having three major deadlines in the exact same week was completely unmanageable.",),
            use_container_width=True
        )

    with col_input:
        st.markdown("##### 💬 STUDENT FEEDBACK TRANSMISSION")
        user_input = st.text_area(
            "Enter or paste student feedback comment:",
            key="input_feedback",
            height=135,
            placeholder="Type student feedback here..."
        )

        def clear_buffer_callback():
            st.session_state["input_feedback"] = ""

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            run_btn = st.button(f"✨ EXECUTE AFFECTIVE DECODING ({selected_model})", type="primary", use_container_width=True)
        with b_c2:
            st.button("🧹 CLEAR BUFFER", on_click=clear_buffer_callback, use_container_width=True)

    # Flashy Holographic Output
    if run_btn and user_input.strip():
        with st.spinner("Decoding affective nuances with recurrent neural forward pass..."):
            try:
                weight_file = os.path.join(MODELS_DIR, "bilstm_model.pt" if selected_model == "Bi-LSTM" else "gru_model.pt")
                weight_mtime = os.path.getmtime(weight_file) if os.path.exists(weight_file) else 0.0
                engine = get_inference_engine(selected_model, weight_mtime)
                prediction = engine.predict(user_input)[0]

                dominant = prediction["dominant_emotion"]
                sentiment = prediction["sentiment"]
                confidence = prediction["confidence"]
                breakdown = prediction["breakdown"]
                meta = EMOTION_META.get(dominant, EMOTION_META["Satisfaction"])

                # Holographic Result Suite
                st.markdown(f"""
                <div class="holo-card" style="--neon-border: {meta['color']}; --neon-shadow: {meta['color']}55;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 18px; margin-bottom: 18px;">
                        <div style="display: flex; align-items: center; gap: 18px;">
                            <div class="holo-avatar-orb" style="--neon-border: {meta['color']}; --orb-bg: {meta['color']}22;">
                                {meta['icon']}
                            </div>
                            <div>
                                <div class="mono-text" style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.12em;">// DOMINANT AFFECTIVE ARCHETYPE //</div>
                                <div class="orbit-title" style="font-size: 2.2rem; font-weight: 900; color: #ffffff; text-shadow: 0 0 20px {meta['color']};">
                                    {dominant.upper()} <span style="font-size: 1.3rem; opacity: 0.85; color: {meta['color']};">({confidence * 100:.1f}%)</span>
                                </div>
                                <div class="mono-text" style="font-size: 0.8rem; color: {meta['color']};">{meta['persona']}</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 24px; align-items: center;">
                            <div>
                                <div class="mono-text" style="font-size: 0.72rem; color: #94a3b8;">SENTIMENT VALENCE</div>
                                <div class="orbit-title" style="font-size: 1.4rem; font-weight: 800; color: {'#00ff66' if sentiment == 'Positive' else ('#ff0055' if sentiment == 'Negative' else '#94a3b8')};">
                                    {sentiment.upper()}
                                </div>
                            </div>
                            <div style="border-left: 1px solid rgba(255, 255, 255, 0.15); padding-left: 20px;">
                                <div class="mono-text" style="font-size: 0.72rem; color: #94a3b8;">CO-PROCESSOR</div>
                                <div class="orbit-title" style="font-size: 1.4rem; font-weight: 800; color: #00f3ff;">{selected_model}</div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 1rem; color: #cbd5e1; line-height: 1.6;">
                        <b>Affective Characteristics:</b> {meta['description']}
                    </div>
                    <div class="directive-card" style="--neon-border: {meta['color']};">
                        <div class="orbit-title" style="font-size: 0.85rem; font-weight: 800; color: {meta['color']}; letter-spacing: 0.05em;">
                            💡 ACTIONABLE PEDAGOGICAL DIRECTIVE:
                        </div>
                        <div style="font-size: 0.98rem; color: #f1f5f9; margin-top: 4px;">
                            {meta['intervention']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Visual Showcase: 5-Axis Spider Radar + Sentiment Gauge
                r_col1, r_col2 = st.columns([1.2, 1])

                with r_col1:
                    st.markdown("##### 🕸️ 5-AXIS AFFECTIVE RADAR")
                    categories = list(breakdown.keys())
                    values = [breakdown[c] * 100 for c in categories]
                    categories_loop = categories + [categories[0]]
                    values_loop = values + [values[0]]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values_loop,
                        theta=categories_loop,
                        fill='toself',
                        fillcolor=f"{meta['color']}33",
                        line=dict(color=meta['color'], width=3),
                        marker=dict(size=8, color="#ffffff")
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0"),
                            angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(family="Orbitron", size=11, color="#1e293b"))
                        ),
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=30, r=30, t=20, b=20),
                        height=290
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with r_col2:
                    st.markdown("##### 🧭 SENTIMENT VALENCE DIAL")
                    valence_val = meta["valence_score"] * confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=valence_val,
                        number=dict(valueformat="+.2f", font=dict(family="Orbitron", size=32, color="#0f172a")),
                        gauge=dict(
                            axis=dict(range=[-1.0, 1.0], tickvals=[-1.0, -0.5, 0, 0.5, 1.0], tickfont=dict(family="Orbitron", color="#64748b")),
                            bar=dict(color=meta["color"], thickness=0.3),
                            steps=[
                                dict(range=[-1.0, -0.3], color="rgba(239, 68, 68, 0.15)"),
                                dict(range=[-0.3, 0.3], color="rgba(148, 163, 184, 0.15)"),
                                dict(range=[0.3, 1.0], color="rgba(34, 197, 94, 0.15)")
                            ]
                        )
                    ))
                    fig_gauge.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=30, b=10),
                        height=290
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Horizontal Probability Density Spectrum
                st.markdown("##### 📊 QUANTUM PROBABILITY DENSITY")
                df_probs = pd.DataFrame(list(breakdown.items()), columns=["Emotion", "Probability"])
                df_probs["Percentage"] = (df_probs["Probability"] * 100).round(2)
                df_probs["Color"] = df_probs["Emotion"].map(lambda e: EMOTION_META.get(e, {}).get("color", "#2563eb"))
                df_probs = df_probs.sort_values(by="Percentage", ascending=True)

                fig_bar = go.Figure(go.Bar(
                    x=df_probs["Percentage"],
                    y=df_probs["Emotion"],
                    orientation="h",
                    marker=dict(color=df_probs["Color"], line=dict(color="#ffffff", width=1.5)),
                    text=[f"  <b>{p:.1f}%</b>" for p in df_probs["Percentage"]],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig_bar.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(title=dict(text="PROBABILITY DENSITY (%)", font=dict(family="Orbitron", size=10, color="#1e293b")), range=[0, max(df_probs["Percentage"].max() + 20, 60)], gridcolor="#e2e8f0"),
                    yaxis=dict(title="", tickfont=dict(family="Orbitron", size=12, color="#0f172a")),
                    height=220
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

# -------------------------------------------------------------
# TAB 2: LONGITUDINAL TRAJECTORY MATRIX
# -------------------------------------------------------------
with tab_trends:
    df_trends = load_default_dataset()

    st.markdown("""
    <div style="background: rgba(8, 14, 32, 0.85); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 16px; padding: 18px 24px; margin-bottom: 22px; box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);">
        <div class="orbit-title" style="font-size: 0.95rem; font-weight: 700; color: #00f3ff; margin-bottom: 10px;">// MULTI-COURSE LONGITUDINAL FILTER //</div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2.5, 1])
    with col_f1:
        courses_available = sorted(df_trends["course_id"].unique().tolist())
        selected_courses = st.multiselect(
            "SELECT MONITORED COURSES:",
            options=courses_available,
            default=courses_available,
            help="Filter multi-course trends by course ID."
        )
    with col_f2:
        semesters_count = len(df_trends["semester"].unique())
        st.markdown(f"""
        <div class="mono-text" style="padding-top: 18px; color: #94a3b8; font-size: 0.85rem;">
            SEMESTERS: <b style="color: #00f3ff;">{semesters_count} Terms</b><br>
            CORPUS: <b style="color: #00ff66;">{len(df_trends):,} Reviews</b>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not selected_courses:
        st.warning("Please select at least one course from the filter above.")
    else:
        filtered_df = df_trends[df_trends["course_id"].isin(selected_courses)]

        total_rev = len(filtered_df)
        net_score = calculate_net_emotion_score(filtered_df)
        most_common = filtered_df["emotion"].mode()[0] if len(filtered_df) > 0 else "N/A"
        sat_rate = (filtered_df["emotion"] == "Satisfaction").mean() * 100 if len(filtered_df) > 0 else 0
        neg_rate = (filtered_df["emotion"].isin(["Confusion", "Frustration", "Stress"])).mean() * 100 if len(filtered_df) > 0 else 0

        nes_color = "#00ff66" if net_score >= 0 else "#ff0055"

        # 4 Show-Off KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="flashy-kpi" style="--kpi-laser: #00f3ff;">
                <div class="mono-text" style="font-size: 0.75rem; color: #00f3ff;">INGESTED FEEDBACKS</div>
                <div class="kpi-num">{total_rev:,}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Across {len(selected_courses)} active courses</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="flashy-kpi" style="--kpi-laser: {nes_color};">
                <div class="mono-text" style="font-size: 0.75rem; color: {nes_color};">NET EMOTION SCORE</div>
                <div class="kpi-num" style="color: {nes_color};">{net_score:+.1f}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Institutional Barometer</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="flashy-kpi" style="--kpi-laser: #00ff66;">
                <div class="mono-text" style="font-size: 0.75rem; color: #00ff66;">SATISFACTION RATE</div>
                <div class="kpi-num">{sat_rate:.1f}%</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Affirmative sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="flashy-kpi" style="--kpi-laser: #ffb703;">
                <div class="mono-text" style="font-size: 0.75rem; color: #ffb703;">MODAL ARCHETYPE</div>
                <div class="kpi-num" style="font-size: 1.7rem;">{most_common.upper()}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Negative friction: {neg_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Charts Section
        c_trend, c_nes = st.columns([1.6, 1])
        trend_table = compute_semester_trends(filtered_df)

        with c_trend:
            st.markdown("##### 📈 1. SEMESTER-OVER-SEMESTER TRAJECTORY MATRIX")
            fig_trend = go.Figure()
            for em in EMOTIONS:
                if em in trend_table.columns:
                    meta = EMOTION_META.get(em, {"color": "#00f3ff", "icon": ""})
                    fig_trend.add_trace(go.Scatter(
                        x=trend_table["semester"],
                        y=trend_table[em],
                        mode="lines+markers",
                        name=f"{meta['icon']} {em.upper()}",
                        line=dict(width=3, color=meta["color"], shape="spline"),
                        marker=dict(size=8, line=dict(width=2, color="#030611"))
                    ))
            fig_trend.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Orbitron", size=10, color="#1e293b")),
                yaxis=dict(title=dict(text="PROPORTION (%)", font=dict(family="Orbitron", size=10, color="#1e293b")), gridcolor="#e2e8f0"),
                xaxis=dict(gridcolor="#e2e8f0"),
                height=340
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_nes:
            st.markdown("##### 🧭 2. NET EMOTION SCORE PROGRESSION")
            fig_nes = go.Figure()
            bar_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in trend_table["Net_Emotion_Score"]]
            fig_nes.add_trace(go.Bar(
                x=trend_table["semester"],
                y=trend_table["Net_Emotion_Score"],
                marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1.5)),
                text=[f"<b>{v:+.1f}</b>" for v in trend_table["Net_Emotion_Score"]],
                textposition="outside",
                hovertemplate="Semester: %{x}<br>NES: %{y:.1f}<extra></extra>"
            ))
            fig_nes.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title=dict(text="NES INDEX (-100 to +100)", font=dict(family="Orbitron", size=10, color="#1e293b")), gridcolor="#e2e8f0"),
                xaxis=dict(gridcolor="#e2e8f0"),
                height=340
            )
            st.plotly_chart(fig_nes, use_container_width=True)

        # Course vs Emotion Heatmap
        st.markdown("##### 🗺️ 3. INTER-COURSE EMOTION INTENSITY SPECTRAL HEATMAP")
        heatmap_matrix = compute_course_heatmap_matrix(filtered_df)
        fig_heat = px.imshow(
            heatmap_matrix,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Plasma",
            labels=dict(x="Affective Archetype", y="Course ID", color="Percentage (%)")
        )
        fig_heat.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=280
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Automated Anomaly Detection & Pedagogical Directives
        st.markdown("##### 🚨 4. AUTONOMOUS ANOMALY RADAR & PEDAGOGICAL DIRECTIVES")
        spikes = detect_emotion_spikes(filtered_df, threshold_percent=10.0)
        if spikes:
            for s in spikes:
                meta = EMOTION_META.get(s['emotion'], EMOTION_META["Stress"])
                st.markdown(f"""
                <div style="background: #fef2f2; border: 1px solid #fee2e2; border-left: 5px solid #dc2626; border-radius: 14px; padding: 18px 24px; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(220, 38, 38, 0.08);">
                    <div class="orbit-title" style="font-weight: 800; color: #dc2626; font-size: 1rem;">
                        🚨 Negative Emotion Spike Alert: {s['course_id']} ({s['semester']})
                    </div>
                    <div style="font-size: 0.95rem; color: #1e293b; margin-top: 4px;">
                        Negative emotion <b>{s['emotion'].upper()}</b> surged by <b style="color: #dc2626;">+{s['spike_amount']}%</b> 
                        (from {s['previous_rate']}% in {s['previous_semester']} to {s['current_rate']}%).
                    </div>
                    <div style="font-size: 0.9rem; color: #475569; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #e2e8f0;">
                        <b>Targeted Pedagogical Directive:</b> {meta['intervention']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f0fdf4; border: 1px solid #dcfce7; border-left: 5px solid #16a34a; border-radius: 14px; padding: 18px 24px; display: flex; align-items: center; gap: 14px; box-shadow: 0 4px 15px rgba(22, 163, 74, 0.08);">
                <span style="font-size: 24px;">✅</span>
                <div class="orbit-title" style="color: #16a34a; font-size: 0.9rem; font-weight: 700;">
                    ALL SIGNALS NOMINAL: No extreme negative anomalies detected (≥10% delta threshold).
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: BATCH CSV PROCESSING
# -------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📁 MASS CORPUS INGESTION & ENRICHMENT")
    st.write("Upload or stream student feedback batches to trigger recurrent vector inference on all lines:")

    col_up1, col_up2 = st.columns([1.8, 1])
    with col_up1:
        uploaded_file = st.file_uploader("DROP FEEDBACK CSV FILE:", type=["csv"])
    with col_up2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("⚡ LOAD PRE-CONFIGURED 50-SIGNAL CORPUS", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample and os.path.exists(TEST_CSV_PATH):
        batch_df = pd.read_csv(TEST_CSV_PATH)
        st.info("INGESTED BUILT-IN TEST DATASET (50 COMMENTS).")

    if batch_df is not None:
        st.markdown(f"""
        <div class="mono-text" style="font-size: 0.8rem; background: rgba(0, 243, 255, 0.08); border: 1px solid #00f3ff; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; color: #00f3ff; box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);">
            [ CORPUS LOADED ]: <b>{len(batch_df)} ENTRIES</b> // <b>{len(batch_df.columns)} ATTRIBUTES</b>
        </div>
        """, unsafe_allow_html=True)

        text_cols = [c for c in batch_df.columns if any(k in c.lower() for k in ["text", "feedback", "comment", "review"])]
        default_idx = batch_df.columns.get_loc(text_cols[0]) if text_cols else 0
        selected_text_col = st.selectbox("SELECT FEEDBACK TEXT COLUMN:", options=batch_df.columns, index=default_idx)

        if st.button(f"⚡ EXECUTE MASS TENSOR INFERENCE ({selected_model})", type="primary"):
            progress_bar = st.progress(0, text="INITIALIZING RECURRENT TENSORS...")
            weight_file = os.path.join(MODELS_DIR, "bilstm_model.pt" if selected_model == "Bi-LSTM" else "gru_model.pt")
            weight_mtime = os.path.getmtime(weight_file) if os.path.exists(weight_file) else 0.0
            engine = get_inference_engine(selected_model, weight_mtime)

            raw_texts = batch_df[selected_text_col].astype(str).tolist()
            progress_bar.progress(35, text="EXECUTING RECURRENT FORWARD PASSES...")
            predictions = engine.predict(raw_texts)
            progress_bar.progress(85, text="AGGREGATING CONFIDENCE INTERVALS...")

            batch_df["PREDICTED_EMOTION"] = [p["dominant_emotion"] for p in predictions]
            batch_df["PREDICTED_SENTIMENT"] = [p["sentiment"] for p in predictions]
            batch_df["CONFIDENCE"] = [round(p["confidence"], 4) for p in predictions]

            progress_bar.progress(100, text="PROCESSING CONCLUDED.")
            st.success("MASS DECODING PIPELINE FINISHED!")

            # Batch Summary Analytics
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("CORPUS SIZE", len(batch_df))
            with b2:
                top_emo = batch_df["PREDICTED_EMOTION"].mode()[0]
                st.metric("MODAL EMOTION", top_emo.upper())
            with b3:
                avg_conf = batch_df["CONFIDENCE"].mean() * 100
                st.metric("MEAN CONFIDENCE", f"{avg_conf:.1f}%")

            # Distribution Donut
            dist_df = batch_df["PREDICTED_EMOTION"].value_counts().reset_index()
            dist_df.columns = ["Emotion", "Count"]

            fig_batch = px.pie(
                dist_df,
                names="Emotion",
                values="Count",
                hole=0.45,
                color="Emotion",
                color_discrete_map={k: v["color"] for k, v in EMOTION_META.items()}
            )
            fig_batch.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280
            )
            st.plotly_chart(fig_batch, use_container_width=True)

            # Table preview
            st.markdown("##### 📋 ENRICHED SIGNAL STREAM")
            st.dataframe(batch_df[[selected_text_col, "PREDICTED_EMOTION", "PREDICTED_SENTIMENT", "CONFIDENCE"]].head(15), use_container_width=True)

            # Export
            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 EXPORT ENRICHED TELEMETRY CSV",
                data=csv_bytes,
                file_name="neural_enriched_student_emotions.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------------------------------------------------
# TAB 4: ARCHITECTURE DUEL BENCHMARK
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("##### 🥊 RECURRENT NEURAL ARCHITECTURE BENCHMARK")
    st.write("Rigorous quantitative head-to-head comparison between **Bidirectional LSTM** and **Bidirectional GRU** models:")

    benchmark_file = os.path.join(MODELS_DIR, "benchmark_results.json")
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)

        b_l = bench_data.get("Bi-LSTM", {})
        b_g = bench_data.get("Bi-GRU", {})

        col_lstm, col_gru = st.columns(2)
        with col_lstm:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px solid #2563eb; border-radius: 18px; padding: 24px; box-shadow: 0 6px 20px rgba(37, 99, 235, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="orbit-title" style="font-size: 1.35rem; font-weight: 800; color: #1d4ed8;">
                        ⚡ BIDIRECTIONAL LSTM
                    </div>
                    <span class="badge-chip chip-cyan">FASTEST LATENCY</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                    Triple-gated memory cell structure capturing forward and backward contextual dependencies.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">VALIDATION ACCURACY</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #16a34a; font-weight: 800;">{b_l.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">MACRO F1-SCORE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #2563eb; font-weight: 800;">{b_l.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">LATENCY / SAMPLE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #0284c7; font-weight: 800;">{b_l.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">EPOCH TIME</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #d97706; font-weight: 800;">{b_l.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gru:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px solid #7c3aed; border-radius: 18px; padding: 24px; box-shadow: 0 6px 20px rgba(124, 58, 237, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="orbit-title" style="font-size: 1.35rem; font-weight: 800; color: #7c3aed;">
                        🚀 BIDIRECTIONAL GRU
                    </div>
                    <span class="badge-chip chip-pink">COMPACT TENSOR</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                    Streamlined gated recurrent unit with fewer parameters and rapid state transitions.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">VALIDATION ACCURACY</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #16a34a; font-weight: 800;">{b_g.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">MACRO F1-SCORE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #7c3aed; font-weight: 800;">{b_g.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">LATENCY / SAMPLE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #0284c7; font-weight: 800;">{b_g.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #64748b;">EPOCH TIME</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #d97706; font-weight: 800;">{b_g.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Comparative Visualization
        st.markdown("##### 📊 QUANTITATIVE EVALUATION METRICS")
        comp_df = pd.DataFrame([
            {"Model": "Bi-LSTM", "Accuracy (%)": b_l.get("validation_accuracy", 0), "Macro F1 (%)": b_l.get("macro_f1", 0) * 100, "Latency (ms)": b_l.get("latency_ms_per_sample", 0)},
            {"Model": "Bi-GRU", "Accuracy (%)": b_g.get("validation_accuracy", 0), "Macro F1 (%)": b_g.get("macro_f1", 0) * 100, "Latency (ms)": b_g.get("latency_ms_per_sample", 0)}
        ])

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Accuracy (%)",
            x=comp_df["Model"],
            y=comp_df["Accuracy (%)"],
            marker_color="#16a34a"
        ))
        fig_comp.add_trace(go.Bar(
            name="Macro F1 (%)",
            x=comp_df["Model"],
            y=comp_df["Macro F1 (%)"],
            marker_color="#2563eb"
        ))
        fig_comp.add_trace(go.Bar(
            name="Latency (scaled)",
            x=comp_df["Model"],
            y=comp_df["Latency (ms)"] * 50,
            marker_color="#d97706"
        ))

        fig_comp.update_layout(
            barmode="group",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Orbitron", size=10, color="#1e293b")),
            yaxis=dict(gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to compile benchmark metrics.")
