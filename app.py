"""
Streamlit Web Application: AI-Based Student Feedback Emotion Analyzer
HYPER-FUTURISTIC AI NEURAL HUD / CYBER-SYNAPSE CORE INTERFACE
"""

import os
import json
import sys
import streamlit as st
import pandas as pd
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
    page_title="AI-CORE // STUDENT EMOTION NEURAL DECODER",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# HYPER-FUTURISTIC AI HUD CYBERNETIC CSS SYSTEM
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

    /* Global AI Font & Deep Space Canvas */
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }

    /* Ambient Cyber Grid + Dark Matter Base */
    .stApp {
        background-color: #030712;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(0, 243, 255, 0.12) 0%, transparent 60%),
            radial-gradient(circle at 100% 100%, rgba(188, 19, 254, 0.1) 0%, transparent 50%),
            linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
        color: #e2e8f0;
    }

    /* Scanline AI Overlay effect */
    .stApp::after {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.35) 50%);
        background-size: 100% 4px;
        z-index: 999999;
        pointer-events: none;
        opacity: 0.45;
    }

    /* HUD Top Terminal Bar */
    .hud-telemetry-bar {
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(3, 10, 24, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.25);
        border-radius: 8px;
        padding: 8px 18px;
        font-size: 0.75rem;
        color: #00f3ff;
        letter-spacing: 0.08em;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.15), inset 0 0 10px rgba(0, 243, 255, 0.05);
    }

    .blinking-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff66;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff66;
        animation: blink 1.2s infinite;
        margin-right: 6px;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }

    /* Cyber Hero HUD Container */
    .cyber-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(8, 18, 38, 0.9) 0%, rgba(3, 7, 18, 0.95) 100%);
        border: 1px solid #00f3ff;
        border-left: 6px solid #00f3ff;
        border-radius: 12px;
        padding: 28px 34px;
        margin-bottom: 26px;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.18), inset 0 0 20px rgba(0, 243, 255, 0.06);
        clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
    }

    .cyber-hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(0, 243, 255, 0.8), 0 0 24px rgba(0, 243, 255, 0.4);
        margin: 0;
        line-height: 1.15;
    }

    .cyber-hero-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.04em;
        margin-top: 10px;
        max-width: 900px;
    }

    .hud-tag-cluster {
        display: flex;
        gap: 12px;
        margin-top: 18px;
        flex-wrap: wrap;
    }

    .hud-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 5px 12px;
        border-radius: 4px;
        background: rgba(0, 243, 255, 0.08);
        border: 1px solid rgba(0, 243, 255, 0.35);
        color: #00f3ff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 0 8px rgba(0, 243, 255, 0.2);
    }

    .hud-badge-pink {
        background: rgba(236, 72, 153, 0.08);
        border-color: rgba(236, 72, 153, 0.4);
        color: #f472b6;
        box-shadow: 0 0 8px rgba(236, 72, 153, 0.2);
    }

    .hud-badge-green {
        background: rgba(0, 255, 102, 0.08);
        border-color: rgba(0, 255, 102, 0.4);
        color: #00ff66;
        box-shadow: 0 0 8px rgba(0, 255, 102, 0.2);
    }

    /* AI Telemetry Metric Box */
    .ai-metric-panel {
        background: rgba(6, 13, 29, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.25);
        border-radius: 10px;
        padding: 18px 20px;
        position: relative;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.08);
        overflow: hidden;
    }

    .ai-metric-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: var(--glow-color, #00f3ff);
        box-shadow: 0 0 10px var(--glow-color, #00f3ff);
    }

    .ai-metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .ai-metric-val {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }

    .ai-metric-sub {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.82rem;
        color: #94a3b8;
        letter-spacing: 0.03em;
        margin-top: 4px;
    }

    /* Synthetic Result Hologram Box */
    .holo-result-box {
        background: linear-gradient(135deg, rgba(8, 22, 45, 0.9) 0%, rgba(3, 8, 20, 0.95) 100%);
        border: 1px solid #00f3ff;
        border-radius: 14px;
        padding: 26px 30px;
        margin-top: 20px;
        margin-bottom: 24px;
        box-shadow: 0 0 35px rgba(0, 243, 255, 0.25), inset 0 0 20px rgba(0, 243, 255, 0.08);
        position: relative;
    }

    .holo-result-box::after {
        content: "[ NEURAL_DECODED // VECTOR_99 ]";
        position: absolute;
        bottom: 8px; right: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: rgba(0, 243, 255, 0.5);
        letter-spacing: 0.12em;
    }

    .holo-emotion-tag {
        font-family: 'Orbitron', monospace;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 10px 22px;
        border-radius: 8px;
        text-shadow: 0 0 12px currentColor;
    }

    .holo-sat {
        background: rgba(0, 255, 102, 0.12);
        color: #00ff66;
        border: 1px solid #00ff66;
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.3);
    }
    .holo-conf {
        background: rgba(255, 183, 3, 0.12);
        color: #ffb703;
        border: 1px solid #ffb703;
        box-shadow: 0 0 20px rgba(255, 183, 3, 0.3);
    }
    .holo-frust {
        background: rgba(255, 0, 85, 0.12);
        color: #ff0055;
        border: 1px solid #ff0055;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.3);
    }
    .holo-bore {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid #94a3b8;
        box-shadow: 0 0 20px rgba(148, 163, 184, 0.3);
    }
    .holo-stress {
        background: rgba(188, 19, 254, 0.12);
        color: #bc13fe;
        border: 1px solid #bc13fe;
        box-shadow: 0 0 20px rgba(188, 19, 254, 0.3);
    }

    /* Custom Streamlit Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(4, 9, 23, 0.8) !important;
        border: 1px solid rgba(0, 243, 255, 0.2) !important;
        padding: 6px 10px !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #64748b !important;
        border-radius: 6px !important;
        padding: 10px 18px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 243, 255, 0.2) 0%, rgba(188, 19, 254, 0.3) 100%) !important;
        color: #00f3ff !important;
        border: 1px solid #00f3ff !important;
        text-shadow: 0 0 8px #00f3ff !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.3) !important;
    }

    /* Cyber Buttons */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 8px !important;
        background: rgba(0, 243, 255, 0.08) !important;
        border: 1px solid rgba(0, 243, 255, 0.4) !important;
        color: #00f3ff !important;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        background: rgba(0, 243, 255, 0.25) !important;
        border-color: #00f3ff !important;
        box-shadow: 0 0 22px rgba(0, 243, 255, 0.4) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }

    /* Primary Accent Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00f3ff 0%, #bc13fe 100%) !important;
        border: none !important;
        color: #020617 !important;
        font-weight: 800 !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.5) !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 35px rgba(188, 19, 254, 0.7) !important;
        color: #ffffff !important;
    }

    /* Cyber Input Area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(4, 9, 23, 0.9) !important;
        border: 1px solid rgba(0, 243, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #38bdf8 !important;
        font-size: 0.95rem !important;
        box-shadow: inset 0 0 15px rgba(0, 243, 255, 0.06) !important;
    }
    .stTextArea textarea:focus {
        border-color: #00f3ff !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.35) !important;
    }

    /* Cyber Card */
    .cyber-card {
        background: rgba(6, 13, 29, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.08);
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
        generate_dataset(n_samples=1800, output_csv=DATA_PATH)
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_inference_engine(model_type: str):
    return EmotionInferenceEngine(model_type=model_type, models_dir=MODELS_DIR)

EMOTION_META = {
    "Satisfaction": {"color": "#00ff66", "holo_class": "holo-sat", "icon": "◈", "vector_label": "AFFIRMATIVE // POSITIVE_VALENCE"},
    "Confusion":    {"color": "#ffb703", "holo_class": "holo-conf", "icon": "▲", "vector_label": "AMBIGUITY // COGNITIVE_DISSONANCE"},
    "Frustration":  {"color": "#ff0055", "holo_class": "holo-frust", "icon": "✖", "vector_label": "IMPEDIMENT // FRICTION_IMPULSE"},
    "Boredom":      {"color": "#94a3b8", "holo_class": "holo-bore", "icon": "—", "vector_label": "MONOTONY // SENSORY_ATTENUATION"},
    "Stress":       {"color": "#bc13fe", "holo_class": "holo-stress", "icon": "⚡", "vector_label": "OVERLOAD // SYNAPTIC_SATURATION"}
}

# -------------------------------------------------------------
# SIDEBAR: SYNAPSE ARCHITECTURE CONTROL
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="border: 1px solid #00f3ff; background: rgba(0, 243, 255, 0.05); padding: 14px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.15);">
        <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 800; color: #00f3ff; letter-spacing: 0.08em;">
            ⚡ AI-CORE // AGY-V9
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #94a3b8; margin-top: 4px;">
            AUTONOMOUS EMOTION ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧬 NEURAL ARCHITECTURE SELECTOR")
    selected_model = st.radio(
        "ACTIVE INFERENCE TENSOR:",
        ["Bi-LSTM", "Bi-GRU"],
        index=0,
        help="Select between Bidirectional Long Short-Term Memory or Bidirectional Gated Recurrent Unit."
    )

    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: rgba(3, 8, 20, 0.9); border: 1px dashed rgba(0, 243, 255, 0.3); border-radius: 8px; padding: 12px; margin-top: 14px; color: #38bdf8;">
        <div>» TENSOR: <b>{selected_model}</b></div>
        <div>» DIRECTION: <b>BIDIRECTIONAL (FORWARD+BACKWARD)</b></div>
        <div>» HIDDEN CHANNELS: <b>128 x 2 DUAL</b></div>
        <div>» POOLING: <b>GLOBAL MAX SEQUENCE POOL</b></div>
        <div>» QUANTUM SOFTMAX: <b>5 AFFECTIVE MODES</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📊 EMOTION VECTOR SPECTRA")
    for emo, meta in EMOTION_META.items():
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; margin-bottom: 8px;">
            <span style="color: {meta['color']}; font-weight: 700;">{meta['icon']} {emo.upper()}</span>
            <span style="color: #64748b; font-size: 0.65rem;">ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748b;">
        <div>CORE LATENCY: <b style="color: #00ff66;">0.34ms</b></div>
        <div>DEVICE: <b style="color: #00f3ff;">CPU // TENSOR_FLOAT32</b></div>
        <div>SECURITY: <b style="color: #00ff66;">ACTIVE_ISOLATION</b></div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TOP HUD TELEMETRY BAR
# -------------------------------------------------------------
st.markdown(f"""
<div class="hud-telemetry-bar">
    <div>
        <span class="blinking-dot"></span>
        <span>AI-AGENT SYNAPTIC CORE [ONLINE]</span>
    </div>
    <div>ACTIVE MODEL: <span style="color: #ffffff; font-weight: 700;">{selected_model}</span></div>
    <div>PORT: <span style="color: #ffffff;">8501 // LOCALHOST</span></div>
    <div>INFERENCE STATUS: <span style="color: #00ff66; font-weight: 700;">READY</span></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CYBER HERO HUD BANNER
# -------------------------------------------------------------
st.markdown(f"""
<div class="cyber-hero">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00f3ff; letter-spacing: 0.15em; margin-bottom: 6px;">
                // SYSTEM: NEURAL_STUDENT_FEEDBACK_ANALYZER_V2 //
            </div>
            <h1 class="cyber-hero-title">Cognitive Emotion Decoder</h1>
            <p class="cyber-hero-subtitle">
                Autonomous recurrent neural network platform translating raw unstructured human feedback transmissions 
                into continuous multi-dimensional affective vectors, real-time sentiment polarity, and longitudinal institutional trajectories.
            </p>
            <div class="hud-tag-cluster">
                <span class="hud-badge">● DUAL-RECURRENT ARCHITECTURE</span>
                <span class="hud-badge-pink">⚡ {selected_model} DECODER</span>
                <span class="hud-badge-green">◈ 5-DIMENSIONAL AFFECTIVE MAPPING</span>
                <span class="hud-badge">📈 TEMPORAL DRIFT DETECTOR</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PRIMARY SYSTEM TABS
# -------------------------------------------------------------
tab_live, tab_trends, tab_batch, tab_benchmark = st.tabs([
    "⚡ REAL-TIME PROBE",
    "📈 TEMPORAL TRAJECTORY MATRIX",
    "📁 MASS CORPUS INGESTION",
    "⚖️ SYNAPSE BENCHMARK"
])

# -------------------------------------------------------------
# TAB 1: REAL-TIME INFERENCE PROBE
# -------------------------------------------------------------
with tab_live:
    col_input, col_presets = st.columns([1.6, 1])

    with col_presets:
        st.markdown("##### 🧬 SYNTHETIC PROBE SIGNALS")
        st.markdown("<p style='font-size: 0.8rem; color: #64748b; font-family: JetBrains Mono;'>INJECT PRE-COMPILED HUMAN FEEDBACK SIGNALS INTO THE NEURAL TENSOR:</p>", unsafe_allow_html=True)

        if "input_feedback" not in st.session_state:
            st.session_state["input_feedback"] = "The professor explained complex topics with crystal clarity. Loved the hands-on labs!"

        def set_probe(text):
            st.session_state["input_feedback"] = text

        st.button(
            "◈ SIGNAL: HIGH_SATISFACTION (Praise & Clarity)",
            on_click=set_probe,
            args=("The professor explained complex topics with crystal clarity. Loved the hands-on labs!",),
            use_container_width=True
        )
        st.button(
            "▲ SIGNAL: COGNITIVE_CONFUSION (Ambiguity)",
            on_click=set_probe,
            args=("Assignment instructions were vague and contradicted what was said in lecture slides.",),
            use_container_width=True
        )
        st.button(
            "✖ SIGNAL: SYSTEM_FRUSTRATION (Grading Delay)",
            on_click=set_probe,
            args=("Grading was delayed for weeks and the exam had questions never covered in class.",),
            use_container_width=True
        )
        st.button(
            "⚡ SIGNAL: SYNAPTIC_STRESS (Extreme Workload)",
            on_click=set_probe,
            args=("Having three major deadlines in the exact same week was completely unmanageable.",),
            use_container_width=True
        )
        st.button(
            "— SIGNAL: ATTENUATION_BOREDOM (Monotony)",
            on_click=set_probe,
            args=("The lectures were very boring, just reading slides verbatim with zero student interaction.",),
            use_container_width=True
        )

    with col_input:
        st.markdown("##### 📥 RAW HUMAN TRANSMISSION INPUT")
        user_input = st.text_area(
            "TRANSMISSION TEXT:",
            key="input_feedback",
            height=140,
            placeholder="[ AWAITING INPUT VECTOR STREAM... ]"
        )

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            run_btn = st.button(f"⚡ INITIATE DECODING SEQUENCE ({selected_model})", type="primary", use_container_width=True)
        with b_c2:
            if st.button("✖ CLEAR BUFFER", use_container_width=True):
                st.session_state["input_feedback"] = ""
                st.rerun()

    # Neural Inference Calculation
    if run_btn and user_input.strip():
        with st.spinner("PROCESSING FORWARD PASS ACROSS RECURRENT TENSOR..."):
            try:
                engine = get_inference_engine(selected_model)
                prediction = engine.predict(user_input)[0]

                dominant = prediction["dominant_emotion"]
                sentiment = prediction["sentiment"]
                confidence = prediction["confidence"]
                breakdown = prediction["breakdown"]
                meta = EMOTION_META.get(dominant, {"color": "#00f3ff", "holo_class": "holo-sat", "icon": "◈", "vector_label": "DECODED"})

                # Holographic Diagnostic Panel
                st.markdown(f"""
                <div class="holo-result-box">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; letter-spacing: 0.12em;">
                                // PRIMARY AFFECTIVE ATTRACTOR //
                            </div>
                            <div class="holo-emotion-tag {meta['holo_class']}" style="margin-top: 8px;">
                                <span>{meta['icon']}</span>
                                <span>{dominant.upper()}</span>
                                <span style="font-size: 1.1rem; opacity: 0.85;">[{confidence * 100:.2f}%]</span>
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {meta['color']}; margin-top: 8px; letter-spacing: 0.08em;">
                                VECTOR: {meta['vector_label']}
                            </div>
                        </div>
                        <div style="display: flex; gap: 20px; align-items: center;">
                            <div style="text-align: right;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;">SENTIMENT VALENCE</div>
                                <div style="font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-top: 4px;">
                                    {sentiment.upper()}
                                </div>
                            </div>
                            <div style="border-left: 1px solid rgba(0, 243, 255, 0.3); padding-left: 20px; text-align: right;">
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em;">NEURAL MODEL</div>
                                <div style="font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 800; color: #00f3ff; margin-top: 4px;">
                                    {selected_model}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Neon Probability Spectrum
                st.markdown("##### 📊 QUANTUM PROBABILITY DENSITY DISTRIBUTION")

                df_probs = pd.DataFrame(list(breakdown.items()), columns=["Emotion", "Probability"])
                df_probs["Percentage"] = (df_probs["Probability"] * 100).round(2)
                df_probs["Color"] = df_probs["Emotion"].map(lambda e: EMOTION_META.get(e, {}).get("color", "#00f3ff"))
                df_probs = df_probs.sort_values(by="Percentage", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_probs["Percentage"],
                    y=df_probs["Emotion"].apply(lambda x: x.upper()),
                    orientation="h",
                    marker=dict(
                        color=df_probs["Color"],
                        line=dict(color="#ffffff", width=1)
                    ),
                    text=[f"  <b>{p:.2f}%</b>" for p in df_probs["Percentage"]],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
                ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(
                        title=dict(text="POSTERIOR PROBABILITY (%)", font=dict(family="Orbitron", size=10, color="#64748b")),
                        range=[0, max(df_probs["Percentage"].max() + 15, 60)],
                        gridcolor="rgba(0, 243, 255, 0.1)",
                        zerolinecolor="rgba(0, 243, 255, 0.2)"
                    ),
                    yaxis=dict(
                        title="",
                        tickfont=dict(family="Orbitron", size=12, color="#e2e8f0")
                    ),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"FATAL EXCEPTION DURING TENSOR FORWARD PASS: {str(e)}")

# -------------------------------------------------------------
# TAB 2: TEMPORAL TRAJECTORY MATRIX
# -------------------------------------------------------------
with tab_trends:
    df_trends = load_default_dataset()

    st.markdown("""
    <div style="background: rgba(4, 9, 23, 0.85); border: 1px solid rgba(0, 243, 255, 0.25); border-radius: 10px; padding: 16px 22px; margin-bottom: 22px;">
        <div style="font-family: 'Orbitron', monospace; font-size: 0.88rem; font-weight: 700; color: #00f3ff; letter-spacing: 0.08em; margin-bottom: 8px;">
            // MULTI-SEMESTER TEMPORAL SLICE FILTER //
        </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2.5, 1])
    with col_f1:
        courses_available = sorted(df_trends["course_id"].unique().tolist())
        selected_courses = st.multiselect(
            "ACTIVE COURSE ENTITIES:",
            options=courses_available,
            default=courses_available,
            help="Select one or more course streams."
        )
    with col_f2:
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding-top: 14px; color: #94a3b8;">
            INSTITUTIONAL TERMS: <b style="color: #00f3ff;">{len(df_trends['semester'].unique())}</b><br>
            CORPUS TRANSMISSIONS: <b style="color: #00ff66;">{len(df_trends):,}</b>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not selected_courses:
        st.warning("NO TARGET ENTITIES SELECTED. SELECT AT LEAST ONE COURSE.")
    else:
        filtered_df = df_trends[df_trends["course_id"].isin(selected_courses)]

        total_rev = len(filtered_df)
        net_score = calculate_net_emotion_score(filtered_df)
        most_common = filtered_df["emotion"].mode()[0] if len(filtered_df) > 0 else "N/A"
        sat_rate = (filtered_df["emotion"] == "Satisfaction").mean() * 100 if len(filtered_df) > 0 else 0
        neg_rate = (filtered_df["emotion"].isin(["Confusion", "Frustration", "Stress"])).mean() * 100 if len(filtered_df) > 0 else 0

        nes_color = "#00ff66" if net_score >= 0 else "#ff0055"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="ai-metric-panel" style="--glow-color: #00f3ff;">
                <div class="ai-metric-label">Ingested Signals</div>
                <div class="ai-metric-val">{total_rev:,}</div>
                <div class="ai-metric-sub">Across {len(selected_courses)} active courses</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="ai-metric-panel" style="--glow-color: {nes_color};">
                <div class="ai-metric-label">Net Emotion Index</div>
                <div class="ai-metric-val" style="color: {nes_color};">{net_score:+.1f}</div>
                <div class="ai-metric-sub">%Sat - %Negative Pressure</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="ai-metric-panel" style="--glow-color: #00ff66;">
                <div class="ai-metric-label">Satisfaction Rate</div>
                <div class="ai-metric-val">{sat_rate:.1f}%</div>
                <div class="ai-metric-sub">Affirmative learning index</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="ai-metric-panel" style="--glow-color: #ffb703;">
                <div class="ai-metric-label">Dominant Modality</div>
                <div class="ai-metric-val" style="font-size: 1.5rem;">{most_common.upper()}</div>
                <div class="ai-metric-sub">Friction coefficient: {neg_rate:.1f}%</div>
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
                    meta = EMOTION_META.get(em, {"color": "#00f3ff", "icon": "◈"})
                    fig_trend.add_trace(go.Scatter(
                        x=trend_table["semester"],
                        y=trend_table[em],
                        mode="lines+markers",
                        name=f"{meta['icon']} {em.upper()}",
                        line=dict(width=3, color=meta["color"], shape="spline"),
                        marker=dict(size=7, line=dict(width=2, color="#030712"))
                    ))
            fig_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Orbitron", size=10)),
                yaxis=dict(title=dict(text="DENSITY (%)", font=dict(family="Orbitron", size=10)), gridcolor="rgba(0, 243, 255, 0.1)"),
                xaxis=dict(gridcolor="rgba(0, 243, 255, 0.1)"),
                height=340
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_nes:
            st.markdown("##### 🧭 2. NET EMOTION INDEX DELTA")
            fig_nes = go.Figure()
            bar_colors = ["#00ff66" if v >= 0 else "#ff0055" for v in trend_table["Net_Emotion_Score"]]
            fig_nes.add_trace(go.Bar(
                x=trend_table["semester"],
                y=trend_table["Net_Emotion_Score"],
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#ffffff", width=1)
                ),
                text=[f"<b>{v:+.1f}</b>" for v in trend_table["Net_Emotion_Score"]],
                textposition="outside",
                hovertemplate="Semester: %{x}<br>NES: %{y:.1f}<extra></extra>"
            ))
            fig_nes.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title=dict(text="NES INDEX", font=dict(family="Orbitron", size=10)), gridcolor="rgba(0, 243, 255, 0.1)"),
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
            labels=dict(x="Affective Modality", y="Course ID", color="Percentage (%)")
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=280
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Automated Pedagogical Spike Alerts
        st.markdown("##### 🚨 4. AUTONOMOUS ANOMALY & SPIKE DETECTION RADAR")
        spikes = detect_emotion_spikes(filtered_df, threshold_percent=10.0)
        if spikes:
            for s in spikes:
                st.markdown(f"""
                <div style="background: rgba(255, 0, 85, 0.08); border-left: 4px solid #ff0055; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace;">
                    <div style="color: #ff0055; font-weight: 700; font-size: 0.88rem;">
                        [ ANOMALY DETECTED // COURSE: {s['course_id']} // TERM: {s['semester']} ]
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.82rem; margin-top: 4px;">
                        Negative vector <b>{s['emotion'].upper()}</b> surged by <b style="color: #ff0055;">+{s['spike_amount']}%</b> 
                        (Shift: {s['previous_rate']}% in {s['previous_semester']} → {s['current_rate']}%).
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(0, 255, 102, 0.06); border-left: 4px solid #00ff66; border-radius: 8px; padding: 14px 18px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #00ff66;">
                [ STATUS: NOMINAL ] No negative affective surges exceeding delta threshold (≥10%) detected.
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: MASS CORPUS INGESTION
# -------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📁 MASS PARALLEL INGESTION & DECODING")
    st.write("Upload or stream student feedback batches to trigger recurrent vector inference on all lines.")

    col_up1, col_up2 = st.columns([1.8, 1])
    with col_up1:
        uploaded_file = st.file_uploader("DROP CORPUS CSV FILE:", type=["csv"])
    with col_up2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("⚡ LOAD PRE-CONFIGURED 50-SIGNAL CORPUS", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample and os.path.exists(TEST_CSV_PATH):
        batch_df = pd.read_csv(TEST_CSV_PATH)
        st.info("INGESTED BUILT-IN TEST DATASET (50 SIGNALS).")

    if batch_df is not None:
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; background: rgba(0, 243, 255, 0.06); border: 1px solid rgba(0, 243, 255, 0.25); border-radius: 6px; padding: 10px 16px; margin-bottom: 14px; color: #00f3ff;">
            [ CORPUS INGESTED ]: <b>{len(batch_df)} ENTRIES</b> // <b>{len(batch_df.columns)} ATTRIBUTES</b>
        </div>
        """, unsafe_allow_html=True)

        text_cols = [c for c in batch_df.columns if any(k in c.lower() for k in ["text", "feedback", "comment", "review"])]
        default_idx = batch_df.columns.get_loc(text_cols[0]) if text_cols else 0
        selected_text_col = st.selectbox("FEEDBACK TEXT COLUMN:", options=batch_df.columns, index=default_idx)

        if st.button(f"⚡ RUN MASS PARALLEL TENSOR INFERENCE ({selected_model})", type="primary"):
            progress_bar = st.progress(0, text="INITIALIZING RECURRENT TENSORS...")
            engine = get_inference_engine(selected_model)

            raw_texts = batch_df[selected_text_col].astype(str).tolist()
            progress_bar.progress(35, text="EXECUTING RECURRENT FORWARD PASSES...")
            predictions = engine.predict(raw_texts)
            progress_bar.progress(85, text="COMPUTING MARGINAL POSTERIORS...")

            batch_df["PREDICTED_EMOTION"] = [p["dominant_emotion"] for p in predictions]
            batch_df["PREDICTED_SENTIMENT"] = [p["sentiment"] for p in predictions]
            batch_df["CONFIDENCE"] = [round(p["confidence"], 4) for p in predictions]

            progress_bar.progress(100, text="PROCESSING CONCLUDED.")
            st.success("MASS DECODING PIPELINE FINISHED.")

            # Analytics row
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("CORPUS SIZE", len(batch_df))
            with b2:
                top_emo = batch_df["PREDICTED_EMOTION"].mode()[0]
                st.metric("MODAL EMOTION", top_emo.upper())
            with b3:
                avg_conf = batch_df["CONFIDENCE"].mean() * 100
                st.metric("MEAN CONFIDENCE", f"{avg_conf:.1f}%")

            # Pie composition
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
# TAB 4: SYNAPSE BENCHMARK
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("##### ⚖️ SYNAPTIC ARCHITECTURE COMBAT: BI-LSTM vs. BI-GRU")
    st.write("Rigorous quantitative tensor evaluation benchmarking parameters, convergence speed, and latency.")

    benchmark_file = os.path.join(MODELS_DIR, "benchmark_results.json")
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)

        b_l = bench_data.get("Bi-LSTM", {})
        b_g = bench_data.get("Bi-GRU", {})

        col_lstm, col_gru = st.columns(2)
        with col_lstm:
            st.markdown(f"""
            <div style="background: rgba(3, 10, 24, 0.9); border: 1px solid #00f3ff; border-top: 4px solid #00f3ff; border-radius: 10px; padding: 22px; box-shadow: 0 0 20px rgba(0, 243, 255, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'Orbitron', monospace; font-size: 1.25rem; font-weight: 800; color: #00f3ff;">
                        ⚡ BIDIRECTIONAL LSTM
                    </div>
                    <span class="hud-badge">FASTEST INFERENCE</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; margin-top: 6px;">
                    Triple-gated memory cell (input, forget, output gates) capturing extended context dependencies.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(0, 243, 255, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(0, 243, 255, 0.15);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">VALIDATION ACCURACY</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #00ff66; font-weight: 800;">{b_l.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(0, 243, 255, 0.15);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">MACRO F1-SCORE</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #00f3ff; font-weight: 800;">{b_l.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(0, 243, 255, 0.15);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">PER-SAMPLE LATENCY</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #38bdf8; font-weight: 800;">{b_l.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(0, 243, 255, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(0, 243, 255, 0.15);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">EPOCH TIME</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #ffb703; font-weight: 800;">{b_l.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gru:
            st.markdown(f"""
            <div style="background: rgba(3, 10, 24, 0.9); border: 1px solid #bc13fe; border-top: 4px solid #bc13fe; border-radius: 10px; padding: 22px; box-shadow: 0 0 20px rgba(188, 19, 254, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-family: 'Orbitron', monospace; font-size: 1.25rem; font-weight: 800; color: #bc13fe;">
                        🚀 BIDIRECTIONAL GRU
                    </div>
                    <span class="hud-badge-pink">COMPACT TENSOR</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; margin-top: 6px;">
                    Dual-gated streamlined recurrent cell (reset & update gates) with minimized parameter surface.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(188, 19, 254, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(188, 19, 254, 0.2);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">VALIDATION ACCURACY</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #00ff66; font-weight: 800;">{b_g.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(188, 19, 254, 0.2);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">MACRO F1-SCORE</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #bc13fe; font-weight: 800;">{b_g.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(188, 19, 254, 0.2);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">PER-SAMPLE LATENCY</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #38bdf8; font-weight: 800;">{b_g.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(188, 19, 254, 0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(188, 19, 254, 0.2);">
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono';">EPOCH TIME</div>
                        <div style="font-family: 'Orbitron'; font-size: 1.4rem; color: #ffb703; font-weight: 800;">{b_g.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Comparative Visualization
        st.markdown("##### 📊 QUANTITATIVE ARCHITECTURE VECTOR COMPARISON")
        comp_df = pd.DataFrame([
            {"Model": "Bi-LSTM", "Accuracy (%)": b_l.get("validation_accuracy", 0), "Macro F1 (%)": b_l.get("macro_f1", 0) * 100, "Latency (ms)": b_l.get("latency_ms_per_sample", 0)},
            {"Model": "Bi-GRU", "Accuracy (%)": b_g.get("validation_accuracy", 0), "Macro F1 (%)": b_g.get("macro_f1", 0) * 100, "Latency (ms)": b_g.get("latency_ms_per_sample", 0)}
        ])

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="ACCURACY (%)",
            x=comp_df["Model"],
            y=comp_df["Accuracy (%)"],
            marker_color="#00ff66"
        ))
        fig_comp.add_trace(go.Bar(
            name="MACRO F1 (%)",
            x=comp_df["Model"],
            y=comp_df["Macro F1 (%)"],
            marker_color="#00f3ff"
        ))
        fig_comp.add_trace(go.Bar(
            name="LATENCY (x50 ms)",
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
        st.info("EXECUTE `python src/train.py` TO COMPILE FORMAL BENCHMARK TELEMETRY.")
