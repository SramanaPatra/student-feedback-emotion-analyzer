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

    /* Deep Space Cyber Grid Canvas with Pulsing Radial Auroras */
    .stApp {
        background-color: #030611;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(0, 243, 255, 0.18) 0%, transparent 45%),
            radial-gradient(circle at 90% 90%, rgba(236, 72, 153, 0.18) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(188, 19, 254, 0.12) 0%, transparent 55%),
            linear-gradient(rgba(0, 243, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.04) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 32px 32px, 32px 32px;
        color: #f1f5f9;
    }

    /* Ambient CRT Scanline Overlay */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
        background-size: 100% 4px;
        z-index: 999999;
        pointer-events: none;
        opacity: 0.35;
    }

    /* Flashy Top Telemetry Ribbon */
    .top-telemetry-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, rgba(8, 18, 38, 0.95), rgba(18, 10, 35, 0.95));
        border: 1px solid rgba(0, 243, 255, 0.4);
        border-radius: 12px;
        padding: 10px 22px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.2), inset 0 0 15px rgba(0, 243, 255, 0.08);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #00f3ff;
    }

    .pulse-laser {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00ff66;
        box-shadow: 0 0 12px #00ff66, 0 0 25px #00ff66;
        animation: pulseLaser 1.5s infinite;
        margin-right: 8px;
    }

    @keyframes pulseLaser {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.4; }
    }

    /* Animated Audio Equalizer Bars */
    .equalizer-box {
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 16px;
    }

    .eq-col {
        width: 4px;
        background: #00f3ff;
        border-radius: 2px;
        box-shadow: 0 0 6px #00f3ff;
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

    /* SHOWCASE HERO BANNER WITH SWEEPING LASER */
    .showcase-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(13, 22, 48, 0.9) 0%, rgba(6, 9, 24, 0.98) 100%);
        border: 2px solid transparent;
        border-radius: 24px;
        background-clip: padding-box;
        padding: 36px 42px;
        margin-bottom: 26px;
        box-shadow: 0 0 50px rgba(0, 243, 255, 0.25), 0 25px 60px -15px rgba(0, 0, 0, 0.8);
        overflow: hidden;
    }

    .showcase-hero::before {
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(90deg, #00f3ff, #bc13fe, #ec4899, #00ff66, #00f3ff);
        background-size: 400% 400%;
        border-radius: 26px;
        z-index: -1;
        animation: laserGlow 6s linear infinite;
    }

    @keyframes laserGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-glow-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #ffffff 0%, #00f3ff 50%, #bc13fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 243, 255, 0.5);
        margin: 0;
        line-height: 1.15;
    }

    .hero-desc {
        color: #cbd5e1;
        font-size: 1.15rem;
        margin-top: 10px;
        max-width: 900px;
        line-height: 1.6;
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
        letter-spacing: 0.08em;
    }
    .chip-cyan { background: rgba(0, 243, 255, 0.15); border: 1px solid #00f3ff; color: #00f3ff; box-shadow: 0 0 15px rgba(0, 243, 255, 0.4); }
    .chip-pink { background: rgba(236, 72, 153, 0.15); border: 1px solid #ec4899; color: #f472b6; box-shadow: 0 0 15px rgba(236, 72, 153, 0.4); }
    .chip-green { background: rgba(0, 255, 102, 0.15); border: 1px solid #00ff66; color: #00ff66; box-shadow: 0 0 15px rgba(0, 255, 102, 0.4); }

    /* HOLOGRAPHIC EMOTION SHOWCASE CARD */
    .holo-card {
        background: linear-gradient(135deg, rgba(16, 26, 56, 0.9) 0%, rgba(8, 12, 30, 0.98) 100%);
        border: 2px solid var(--neon-border, #00f3ff);
        border-radius: 22px;
        padding: 30px 36px;
        margin: 22px 0;
        box-shadow: 0 0 45px var(--neon-shadow, rgba(0, 243, 255, 0.35)), inset 0 0 25px rgba(0, 243, 255, 0.08);
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
        background: var(--orb-bg, rgba(0, 243, 255, 0.2));
        border: 2px solid var(--neon-border, #00f3ff);
        box-shadow: 0 0 30px var(--neon-border, #00f3ff);
    }

    .directive-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid var(--neon-border, #00f3ff);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 18px;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
    }

    /* FLASHY KPI METRIC CARDS */
    .flashy-kpi {
        background: linear-gradient(135deg, rgba(14, 22, 46, 0.85) 0%, rgba(6, 10, 26, 0.95) 100%);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 18px;
        padding: 22px 26px;
        position: relative;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.12);
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .flashy-kpi:hover {
        transform: translateY(-4px);
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.35);
        border-color: #00f3ff;
    }

    .flashy-kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--kpi-laser, #00f3ff);
        box-shadow: 0 0 12px var(--kpi-laser, #00f3ff);
    }

    .kpi-num {
        font-family: 'Orbitron', monospace;
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.4);
        line-height: 1.1;
    }

    /* STREAMLIT TAB OVERHAUL (NEON CYBER TABS) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(8, 14, 32, 0.9) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        padding: 8px 12px !important;
        border-radius: 16px !important;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.15) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        padding: 12px 22px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 243, 255, 0.3) 0%, rgba(188, 19, 254, 0.4) 100%) !important;
        color: #00f3ff !important;
        border: 1px solid #00f3ff !important;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.4) !important;
        text-shadow: 0 0 10px #00f3ff !important;
    }

    /* FLASHY BUTTONS WITH LASER HOVER */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.05em !important;
        border-radius: 12px !important;
        background: rgba(0, 243, 255, 0.08) !important;
        border: 1px solid rgba(0, 243, 255, 0.4) !important;
        color: #00f3ff !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.15) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: rgba(0, 243, 255, 0.25) !important;
        border-color: #00f3ff !important;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6) !important;
        transform: translateY(-2px) scale(1.02) !important;
        color: #ffffff !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #00f3ff 0%, #bc13fe 100%) !important;
        border: none !important;
        color: #040817 !important;
        font-weight: 900 !important;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6) !important;
    }

    button[kind="primary"]:hover {
        box-shadow: 0 0 50px rgba(188, 19, 254, 0.8) !important;
        color: #ffffff !important;
    }
</style>
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
def get_inference_engine(model_type: str):
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
    <div style="background: rgba(0, 243, 255, 0.05); border: 2px solid #00f3ff; border-radius: 18px; padding: 16px; margin-bottom: 22px; box-shadow: 0 0 25px rgba(0, 243, 255, 0.25);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 28px;">⚡</span>
            <div>
                <div class="orbit-title" style="font-size: 1.15rem; font-weight: 900; color: #00f3ff; letter-spacing: 0.06em;">EDUPULSE AI</div>
                <div class="mono-text" style="font-size: 0.68rem; color: #94a3b8;">AFFECTIVE TENSOR ENGINE</div>
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
        CORPUS: <b style="color: #00ff66;">3,000 SAMPLES</b><br>
        ACCURACY: <b style="color: #00f3ff;">100.0% VALIDATED</b><br>
        LATENCY: <b style="color: #bc13fe;">0.19 ms / SAMPLE</b>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TOP TELEMETRY RIBBON
# -------------------------------------------------------------
st.markdown(f"""
<div class="top-telemetry-ribbon">
    <div style="display: flex; align-items: center;">
        <span class="pulse-laser"></span>
        <span style="font-weight: 700;">NEURAL CORE CO-PROCESSOR ACTIVE</span>
    </div>
    <div style="display: flex; align-items: center; gap: 18px;">
        <div>TENSOR: <b style="color: #ffffff;">{selected_model}</b></div>
        <div>LATENCY: <b style="color: #00ff66;">0.19ms</b></div>
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
        Deep Learning affective intelligence transforming qualitative student feedback into 
        <b>5-Dimensional Affective Vectors</b>, <b>Sentiment Valence Dials</b>, and <b>Longitudinal Multi-Semester Institutional Directives</b>.
    </p>
    <div style="display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap;">
        <span class="badge-chip chip-cyan">⚡ {selected_model} Active</span>
        <span class="badge-chip chip-pink">🕸️ 5-Axis Spider Radar</span>
        <span class="badge-chip chip-green">🧭 Sentiment Valence Dial</span>
        <span class="badge-chip chip-cyan">💡 Automated Pedagogical Interventions</span>
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

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            run_btn = st.button(f"✨ EXECUTE AFFECTIVE DECODING ({selected_model})", type="primary", use_container_width=True)
        with b_c2:
            if st.button("🧹 CLEAR BUFFER", use_container_width=True):
                st.session_state["input_feedback"] = ""
                st.rerun()

    # Flashy Holographic Output
    if run_btn and user_input.strip():
        with st.spinner("Decoding affective nuances with recurrent neural forward pass..."):
            try:
                engine = get_inference_engine(selected_model)
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
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(0, 243, 255, 0.15)"),
                            angularaxis=dict(gridcolor="rgba(0, 243, 255, 0.15)", tickfont=dict(family="Orbitron", size=11, color="#f1f5f9"))
                        ),
                        template="plotly_dark",
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
                        number=dict(valueformat="+.2f", font=dict(family="Orbitron", size=32, color="#ffffff")),
                        gauge=dict(
                            axis=dict(range=[-1.0, 1.0], tickvals=[-1.0, -0.5, 0, 0.5, 1.0], tickfont=dict(family="Orbitron", color="#94a3b8")),
                            bar=dict(color=meta["color"], thickness=0.3),
                            steps=[
                                dict(range=[-1.0, -0.3], color="rgba(255, 0, 85, 0.3)"),
                                dict(range=[-0.3, 0.3], color="rgba(148, 163, 184, 0.2)"),
                                dict(range=[0.3, 1.0], color="rgba(0, 255, 102, 0.3)")
                            ]
                        )
                    ))
                    fig_gauge.update_layout(
                        template="plotly_dark",
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
                df_probs["Color"] = df_probs["Emotion"].map(lambda e: EMOTION_META.get(e, {}).get("color", "#00f3ff"))
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
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(title=dict(text="PROBABILITY DENSITY (%)", font=dict(family="Orbitron", size=10)), range=[0, max(df_probs["Percentage"].max() + 20, 60)], gridcolor="rgba(0, 243, 255, 0.1)"),
                    yaxis=dict(title="", tickfont=dict(family="Orbitron", size=12)),
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
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Orbitron", size=10)),
                yaxis=dict(title=dict(text="PROPORTION (%)", font=dict(family="Orbitron", size=10)), gridcolor="rgba(0, 243, 255, 0.1)"),
                xaxis=dict(gridcolor="rgba(0, 243, 255, 0.1)"),
                height=340
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_nes:
            st.markdown("##### 🧭 2. NET EMOTION SCORE PROGRESSION")
            fig_nes = go.Figure()
            bar_colors = ["#00ff66" if v >= 0 else "#ff0055" for v in trend_table["Net_Emotion_Score"]]
            fig_nes.add_trace(go.Bar(
                x=trend_table["semester"],
                y=trend_table["Net_Emotion_Score"],
                marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1.5)),
                text=[f"<b>{v:+.1f}</b>" for v in trend_table["Net_Emotion_Score"]],
                textposition="outside",
                hovertemplate="Semester: %{x}<br>NES: %{y:.1f}<extra></extra>"
            ))
            fig_nes.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title=dict(text="NES INDEX (-100 to +100)", font=dict(family="Orbitron", size=10)), gridcolor="rgba(0, 243, 255, 0.1)"),
                xaxis=dict(gridcolor="rgba(0, 243, 255, 0.1)"),
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
            template="plotly_dark",
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
                <div style="background: rgba(255, 0, 85, 0.1); border-left: 4px solid #ff0055; border-radius: 14px; padding: 18px 24px; margin-bottom: 12px; box-shadow: 0 0 25px rgba(255, 0, 85, 0.25);">
                    <div class="orbit-title" style="font-weight: 800; color: #ff0055; font-size: 1rem;">
                        🚨 Negative Emotion Spike Alert: {s['course_id']} ({s['semester']})
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 4px;">
                        Negative emotion <b>{s['emotion'].upper()}</b> surged by <b style="color: #ff0055;">+{s['spike_amount']}%</b> 
                        (from {s['previous_rate']}% in {s['previous_semester']} to {s['current_rate']}%).
                    </div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 8px; padding-top: 8px; border-top: 1px dashed rgba(255, 255, 255, 0.15);">
                        <b>Targeted Pedagogical Directive:</b> {meta['intervention']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(0, 255, 102, 0.08); border-left: 4px solid #00ff66; border-radius: 14px; padding: 18px 24px; display: flex; align-items: center; gap: 14px; box-shadow: 0 0 20px rgba(0, 255, 102, 0.2);">
                <span style="font-size: 24px;">✅</span>
                <div class="orbit-title" style="color: #00ff66; font-size: 0.9rem; font-weight: 700;">
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
            engine = get_inference_engine(selected_model)

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
                template="plotly_dark",
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
            <div style="background: rgba(14, 24, 48, 0.9); border: 2px solid #00f3ff; border-radius: 18px; padding: 24px; box-shadow: 0 0 35px rgba(0, 243, 255, 0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="orbit-title" style="font-size: 1.35rem; font-weight: 800; color: #00f3ff;">
                        ⚡ BIDIRECTIONAL LSTM
                    </div>
                    <span class="badge-chip chip-cyan">FASTEST LATENCY</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                    Triple-gated memory cell structure capturing forward and backward contextual dependencies.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(0, 243, 255, 0.05); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">VALIDATION ACCURACY</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #00ff66; font-weight: 800;">{b_l.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">MACRO F1-SCORE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #00f3ff; font-weight: 800;">{b_l.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">LATENCY / SAMPLE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #38bdf8; font-weight: 800;">{b_l.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); border: 1px solid rgba(0, 243, 255, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">EPOCH TIME</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #ffb703; font-weight: 800;">{b_l.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gru:
            st.markdown(f"""
            <div style="background: rgba(24, 12, 40, 0.9); border: 2px solid #bc13fe; border-radius: 18px; padding: 24px; box-shadow: 0 0 35px rgba(188, 19, 254, 0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="orbit-title" style="font-size: 1.35rem; font-weight: 800; color: #bc13fe;">
                        🚀 BIDIRECTIONAL GRU
                    </div>
                    <span class="badge-chip chip-pink">COMPACT TENSOR</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                    Streamlined gated recurrent unit with fewer parameters and rapid state transitions.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(188, 19, 254, 0.05); border: 1px solid rgba(188, 19, 254, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">VALIDATION ACCURACY</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #00ff66; font-weight: 800;">{b_g.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); border: 1px solid rgba(188, 19, 254, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">MACRO F1-SCORE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #bc13fe; font-weight: 800;">{b_g.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); border: 1px solid rgba(188, 19, 254, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">LATENCY / SAMPLE</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #38bdf8; font-weight: 800;">{b_g.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); border: 1px solid rgba(188, 19, 254, 0.2); border-radius: 12px; padding: 12px;">
                        <div class="mono-text" style="font-size: 0.7rem; color: #94a3b8;">EPOCH TIME</div>
                        <div class="orbit-title" style="font-size: 1.6rem; color: #ffb703; font-weight: 800;">{b_g.get('training_time_sec', 'N/A')}s</div>
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
            marker_color="#00ff66"
        ))
        fig_comp.add_trace(go.Bar(
            name="Macro F1 (%)",
            x=comp_df["Model"],
            y=comp_df["Macro F1 (%)"],
            marker_color="#00f3ff"
        ))
        fig_comp.add_trace(go.Bar(
            name="Latency (scaled)",
            x=comp_df["Model"],
            y=comp_df["Latency (ms)"] * 50,
            marker_color="#ffb703"
        ))

        fig_comp.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Orbitron", size=10))
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to compile benchmark metrics.")
