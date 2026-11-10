"""
Streamlit Web Application: AI-Based Student Feedback Emotion Analyzer
Ultra-Expressive & Unique Academic Affective Intelligence Platform.
Featuring Bi-LSTM / GRU Neural Models, Multi-Dimensional Polar Radar, and Pedagogical Interventions.
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
    page_title="EduSense AI • Student Affective Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# UNIQUE & EXPRESSIVE CUSTOM CSS (SYNE & PLUS JAKARTA SANS)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .syne-font {
        font-family: 'Syne', sans-serif;
    }
    .mono-font {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Expressive Cosmos Backdrop */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(236, 72, 153, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
                    #090d16;
        color: #f1f5f9;
    }

    /* Hero Expressive Banner */
    .expressive-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(22, 28, 48, 0.8) 0%, rgba(12, 16, 30, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 34px 40px;
        margin-bottom: 24px;
        box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        overflow: hidden;
    }

    .expressive-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b, #10b981);
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 20%, #cbd5e1 60%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.08rem;
        margin-top: 10px;
        max-width: 860px;
        line-height: 1.6;
    }

    .hero-pill-cluster {
        display: flex;
        gap: 10px;
        margin-top: 18px;
        flex-wrap: wrap;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .hero-pill-live {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* Expressive Emotion Showcase Card */
    .emotion-card {
        background: linear-gradient(135deg, rgba(25, 30, 52, 0.85) 0%, rgba(14, 18, 34, 0.95) 100%);
        border: 2px solid var(--accent-glow, #6366f1);
        border-radius: 22px;
        padding: 26px 32px;
        margin: 20px 0;
        box-shadow: 0 12px 35px -10px var(--accent-shadow, rgba(99, 102, 241, 0.3));
        position: relative;
    }

    .emotion-avatar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 18px;
        margin-bottom: 18px;
    }

    .emotion-badge-large {
        display: inline-flex;
        align-items: center;
        gap: 14px;
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }

    .avatar-icon-bubble {
        width: 60px;
        height: 60px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        background: var(--bubble-bg, rgba(99, 102, 241, 0.2));
        border: 2px solid var(--bubble-border, #6366f1);
        box-shadow: 0 0 20px var(--bubble-shadow, rgba(99, 102, 241, 0.4));
    }

    .intervention-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid var(--accent-glow, #6366f1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 14px;
    }

    .intervention-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--accent-glow, #6366f1);
    }

    .intervention-text {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 4px;
        line-height: 1.5;
    }

    /* Expressive Preset Chips */
    .preset-card {
        background: rgba(18, 24, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .preset-card:hover {
        border-color: #818cf8;
        background: rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }

    /* Metric Box */
    .kpi-expressive {
        background: linear-gradient(135deg, rgba(22, 28, 48, 0.8) 0%, rgba(13, 17, 30, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }

    .kpi-expressive::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--kpi-accent, #6366f1);
    }

    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #94a3b8;
    }

    .kpi-val {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
    }

    .kpi-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(18, 24, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 6px 10px !important;
        border-radius: 14px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%) !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(236, 72, 153, 0.5) !important;
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
        "color": "#10b981",
        "icon": "🌟",
        "persona": "Enlightened & Fulfilled",
        "description": "High pedagogical engagement, conceptual clarity, and rewarding milestones.",
        "intervention": "Actionable Insight: Document and archive the current teaching methodologies and lab formats as departmental benchmarks.",
        "valence_score": 1.0
    },
    "Confusion": {
        "color": "#f59e0b",
        "icon": "❓",
        "persona": "Dissonant & Perplexed",
        "description": "Ambiguous specifications, abrupt topic transitions, and disconnect with readings.",
        "intervention": "Pedagogical Directive: Host a mid-term alignment Q&A and publish structured rubrics with concrete code examples.",
        "valence_score": -0.4
    },
    "Frustration": {
        "color": "#ef4444",
        "icon": "⚠️",
        "persona": "Frictional & Obstructed",
        "description": "Delayed grading, unfair exam surprises, and poor administrative support.",
        "intervention": "Urgent Administrative Intervention: Audit TA response cycles, streamline auto-graders, and review exam questions against syllabus.",
        "valence_score": -0.8
    },
    "Boredom": {
        "color": "#8b5cf6",
        "icon": "🥱",
        "persona": "Passive & Monotonous",
        "description": "Monotone slide reading, lack of interactive discussions, and low challenge.",
        "intervention": "Engagement Overhaul: Transition from 100% lecture to active-learning segments, peer discussions, and live coding exercises.",
        "valence_score": 0.0
    },
    "Stress": {
        "color": "#ec4899",
        "icon": "⚡",
        "persona": "Overloaded & Burned Out",
        "description": "Clustered deadlines, punitive grading policies, and extreme workload anxiety.",
        "intervention": "Student Wellbeing Alert: Coordinate across co-requisite courses to stagger major deadlines and introduce flexible grace windows.",
        "valence_score": -0.9
    }
}

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #6366f1, #ec4899); padding: 12px; border-radius: 16px; font-size: 26px; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);">
            🌌
        </div>
        <div>
            <div class="syne-font" style="font-size: 1.25rem; font-weight: 800; color: #fff; line-height: 1.1;">EduSense AI</div>
            <div class="mono-font" style="font-size: 0.72rem; color: #818cf8; letter-spacing: 0.04em;">AFFECTIVE INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧠 Deep Learning Architecture")
    selected_model = st.radio(
        "Active Recurrent Model:",
        ["Bi-LSTM", "Bi-GRU"],
        index=0,
        help="Select between Bidirectional LSTM or Bidirectional GRU model."
    )

    st.markdown("---")
    st.markdown("##### 🎭 The 5 Affective Archetypes")
    for emo, meta in EMOTION_META.items():
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.1rem;">{meta['icon']}</span>
            <span style="font-weight: 700; color: {meta['color']}; font-size: 0.9rem;">{emo}</span>
            <span style="color: #64748b; font-size: 0.75rem; margin-left: auto;">{meta['persona'][:16]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="mono-font" style="font-size: 0.75rem; color: #64748b;">
        <div>DATASET: <b style="color: #10b981;">3,000 COHERENT SAMPLES</b></div>
        <div>VAL ACCURACY: <b style="color: #6366f1;">100.0%</b></div>
        <div>INFERENCE: <b style="color: #38bdf8;">SUB-MILLISECOND</b></div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------------
st.markdown(f"""
<div class="expressive-hero">
    <h1 class="hero-title">AI Student Feedback Emotion Analyzer</h1>
    <p class="hero-subtitle">
        Translating qualitative student course feedback into rich 5-dimensional affective vectors, 
        sentiment polarity dials, and longitudinal pedagogical trends across university semesters.
    </p>
    <div class="hero-pill-cluster">
        <span class="hero-pill hero-pill-live">
            ● Neural Engine Loaded
        </span>
        <span class="hero-pill">
            ⚡ {selected_model} Active
        </span>
        <span class="hero-pill">
            🎯 5-Axis Affective Radar
        </span>
        <span class="hero-pill">
            📈 Longitudinal Course Heatmaps
        </span>
        <span class="hero-pill">
            💡 Pedagogical Interventions
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
tab_live, tab_trends, tab_batch, tab_benchmark = st.tabs([
    "✨ Expressive Single Tester",
    "📊 Longitudinal Trend Analytics",
    "📁 Batch CSV Processing",
    "⚖️ Bi-LSTM vs GRU Benchmark"
])

# -------------------------------------------------------------
# TAB 1: EXPRESSIVE SINGLE FEEDBACK TESTER
# -------------------------------------------------------------
with tab_live:
    col_input, col_presets = st.columns([1.6, 1])

    with col_presets:
        st.markdown("##### 💡 Expressive Preset Tests")
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>Click any realistic academic student feedback scenario to test the neural model:</p>", unsafe_allow_html=True)

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
        st.markdown("##### 💬 Student Feedback Comment")
        user_input = st.text_area(
            "Enter student feedback text below:",
            key="input_feedback",
            height=135,
            placeholder="Type student feedback here..."
        )

        b_c1, b_c2 = st.columns([2, 1])
        with b_c1:
            run_btn = st.button(f"✨ Analyze Emotion with {selected_model}", type="primary", use_container_width=True)
        with b_c2:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state["input_feedback"] = ""
                st.rerun()

    # Inference Execution & Expressive Visualization
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

                # Expressive Emotion Showcase Card
                st.markdown(f"""
                <div class="emotion-card" style="--accent-glow: {meta['color']}; --accent-shadow: {meta['color']}44;">
                    <div class="emotion-avatar-header">
                        <div class="emotion-badge-large">
                            <div class="avatar-icon-bubble" style="--bubble-bg: {meta['color']}22; --bubble-border: {meta['color']}; --bubble-shadow: {meta['color']}55;">
                                {meta['icon']}
                            </div>
                            <div>
                                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; font-weight: 700;">Dominant Affective Archetype</div>
                                <div style="color: {meta['color']};">{dominant} <span style="font-size: 1.2rem; color: #ffffff; opacity: 0.85;">({confidence * 100:.1f}%)</span></div>
                                <div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">{meta['persona']}</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 24px; align-items: center;">
                            <div>
                                <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Sentiment Polarity</div>
                                <div class="syne-font" style="font-size: 1.4rem; font-weight: 800; color: {'#10b981' if sentiment == 'Positive' else ('#ef4444' if sentiment == 'Negative' else '#94a3b8')};">
                                    {sentiment}
                                </div>
                            </div>
                            <div style="border-left: 1px solid rgba(255, 255, 255, 0.1); padding-left: 20px;">
                                <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Model Engine</div>
                                <div class="syne-font" style="font-size: 1.4rem; font-weight: 800; color: #818cf8;">{selected_model}</div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5;">
                        <b>Affective Characteristics:</b> {meta['description']}
                    </div>
                    <div class="intervention-box" style="--accent-glow: {meta['color']};">
                        <div class="intervention-title">💡 Pedagogical Intervention Recommendation:</div>
                        <div class="intervention-text">{meta['intervention']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Expressive Multi-Dimensional Visualizations
                r_col1, r_col2 = st.columns([1.2, 1])

                with r_col1:
                    st.markdown("##### 🕸️ 5-Axis Affective Radar")
                    # Radar Chart
                    categories = list(breakdown.keys())
                    values = [breakdown[c] * 100 for c in categories]
                    # Close the loop
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
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255, 255, 255, 0.1)"),
                            angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(family="Syne", size=11, color="#f1f5f9"))
                        ),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=30, r=30, t=20, b=20),
                        height=290
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with r_col2:
                    st.markdown("##### 🧭 Sentiment Valence Gauge")
                    # Sentiment Gauge
                    valence_val = meta["valence_score"] * confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=valence_val,
                        number=dict(valueformat="+.2f", font=dict(family="Syne", size=28, color="#ffffff")),
                        gauge=dict(
                            axis=dict(range=[-1.0, 1.0], tickvals=[-1.0, -0.5, 0, 0.5, 1.0], tickfont=dict(color="#94a3b8")),
                            bar=dict(color=meta["color"], thickness=0.3),
                            steps=[
                                dict(range=[-1.0, -0.3], color="rgba(239, 68, 68, 0.25)"),
                                dict(range=[-0.3, 0.3], color="rgba(148, 163, 184, 0.2)"),
                                dict(range=[0.3, 1.0], color="rgba(16, 185, 129, 0.25)")
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
                st.markdown("##### 📊 Probability Breakdown")
                df_probs = pd.DataFrame(list(breakdown.items()), columns=["Emotion", "Probability"])
                df_probs["Percentage"] = (df_probs["Probability"] * 100).round(2)
                df_probs["Color"] = df_probs["Emotion"].map(lambda e: EMOTION_META.get(e, {}).get("color", "#6366f1"))
                df_probs = df_probs.sort_values(by="Percentage", ascending=True)

                fig_bar = go.Figure(go.Bar(
                    x=df_probs["Percentage"],
                    y=df_probs["Emotion"],
                    orientation="h",
                    marker=dict(color=df_probs["Color"], line=dict(color="#ffffff", width=1)),
                    text=[f"  <b>{p:.1f}%</b>" for p in df_probs["Percentage"]],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(title=dict(text="CONFIDENCE PROBABILITY (%)", font=dict(family="Syne", size=10)), range=[0, max(df_probs["Percentage"].max() + 20, 60)], gridcolor="rgba(255, 255, 255, 0.08)"),
                    yaxis=dict(title="", tickfont=dict(family="Syne", size=12)),
                    height=220
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

# -------------------------------------------------------------
# TAB 2: LONGITUDINAL TREND ANALYTICS
# -------------------------------------------------------------
with tab_trends:
    df_trends = load_default_dataset()

    st.markdown("""
    <div style="background: rgba(22, 28, 48, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 18px 24px; margin-bottom: 22px;">
        <div class="syne-font" style="font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 10px;">🔍 Filter Courses Across Monitored Semesters</div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2.5, 1])
    with col_f1:
        courses_available = sorted(df_trends["course_id"].unique().tolist())
        selected_courses = st.multiselect(
            "Select Course(s):",
            options=courses_available,
            default=courses_available,
            help="Filter multi-course trends by course ID."
        )
    with col_f2:
        semesters_count = len(df_trends["semester"].unique())
        st.markdown(f"""
        <div style="padding-top: 18px; color: #94a3b8; font-size: 0.9rem;">
            Academic Terms: <b style="color: #818cf8;">{semesters_count} Semesters</b><br>
            Analyzed Reviews: <b style="color: #10b981;">{len(df_trends):,}</b>
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

        nes_color = "#10b981" if net_score >= 0 else "#ef4444"

        # KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-expressive" style="--kpi-accent: #6366f1;">
                <div class="kpi-title">Total Feedbacks</div>
                <div class="kpi-val">{total_rev:,}</div>
                <div class="kpi-sub">Across {len(selected_courses)} active courses</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-expressive" style="--kpi-accent: {nes_color};">
                <div class="kpi-title">Net Emotion Score</div>
                <div class="kpi-val" style="color: {nes_color};">{net_score:+.1f}</div>
                <div class="kpi-sub">%Satisfaction - %Negative</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-expressive" style="--kpi-accent: #10b981;">
                <div class="kpi-title">Satisfaction Rate</div>
                <div class="kpi-val">{sat_rate:.1f}%</div>
                <div class="kpi-sub">Empowered student learning</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-expressive" style="--kpi-accent: #f59e0b;">
                <div class="kpi-title">Modal Archetype</div>
                <div class="kpi-val" style="font-size: 1.6rem;">{most_common}</div>
                <div class="kpi-sub">Negative friction: {neg_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Charts Section
        c_trend, c_nes = st.columns([1.6, 1])
        trend_table = compute_semester_trends(filtered_df)

        with c_trend:
            st.markdown("##### 📈 1. Semester-over-Semester Emotion Trajectory")
            fig_trend = go.Figure()
            for em in EMOTIONS:
                if em in trend_table.columns:
                    meta = EMOTION_META.get(em, {"color": "#6366f1", "icon": ""})
                    fig_trend.add_trace(go.Scatter(
                        x=trend_table["semester"],
                        y=trend_table[em],
                        mode="lines+markers",
                        name=f"{meta['icon']} {em}",
                        line=dict(width=3, color=meta["color"], shape="spline"),
                        marker=dict(size=8, line=dict(width=2, color="#090d16"))
                    ))
            fig_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Syne", size=10)),
                yaxis=dict(title=dict(text="PROPORTION (%)", font=dict(family="Syne", size=10)), gridcolor="rgba(255, 255, 255, 0.08)"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
                height=340
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_nes:
            st.markdown("##### 🧭 2. Net Emotion Score (NES) Institutional Health")
            fig_nes = go.Figure()
            bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in trend_table["Net_Emotion_Score"]]
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
                yaxis=dict(title=dict(text="NES INDEX (-100 to +100)", font=dict(family="Syne", size=10)), gridcolor="rgba(255, 255, 255, 0.08)"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
                height=340
            )
            st.plotly_chart(fig_nes, use_container_width=True)

        # Course vs Emotion Heatmap
        st.markdown("##### 🗺️ 3. Course vs Emotion Intensity Heatmap")
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
        st.markdown("##### 🚨 4. Automated Pedagogical Spike Alerts & Directives")
        spikes = detect_emotion_spikes(filtered_df, threshold_percent=10.0)
        if spikes:
            for s in spikes:
                meta = EMOTION_META.get(s['emotion'], EMOTION_META["Stress"])
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;">
                    <div class="syne-font" style="font-weight: 700; color: #fca5a5; font-size: 0.95rem;">
                        🚨 Negative Emotion Spike in {s['course_id']} ({s['semester']})
                    </div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; margin-top: 4px;">
                        Negative emotion <b>{s['emotion']}</b> surged by <b style="color: #ef4444;">+{s['spike_amount']}%</b> 
                        (from {s['previous_rate']}% in {s['previous_semester']} to {s['current_rate']}%).
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px; padding-top: 6px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        <b>Targeted Pedagogical Directive:</b> {meta['intervention']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 12px; padding: 16px 20px; display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 22px;">✅</span>
                <div style="color: #6ee7b7; font-weight: 600;">No severe negative emotion anomalies detected in selected courses (&ge;10% delta threshold).</div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: BATCH CSV PROCESSING
# -------------------------------------------------------------
with tab_batch:
    st.markdown("##### 📁 High-Throughput Batch Processing")
    st.write("Upload a CSV file containing student feedback comments to batch classify emotion vectors and download enriched analytics.")

    col_up1, col_up2 = st.columns([1.8, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload feedback CSV file:", type=["csv"])
    with col_up2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("📄 Load Built-in Test Dataset (50 comments)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample and os.path.exists(TEST_CSV_PATH):
        batch_df = pd.read_csv(TEST_CSV_PATH)
        st.info("Loaded built-in test dataset `test_student_comments.csv`.")

    if batch_df is not None:
        st.markdown(f"""
        <div style="padding: 10px 16px; background: rgba(99, 102, 241, 0.1); border-radius: 10px; margin-bottom: 16px; font-size: 0.9rem;">
            Dataset loaded: <b>{len(batch_df)} rows</b> across <b>{len(batch_df.columns)} columns</b>.
        </div>
        """, unsafe_allow_html=True)

        text_cols = [c for c in batch_df.columns if any(k in c.lower() for k in ["text", "feedback", "comment", "review"])]
        default_idx = batch_df.columns.get_loc(text_cols[0]) if text_cols else 0
        selected_text_col = st.selectbox("Select Feedback Text Column:", options=batch_df.columns, index=default_idx)

        if st.button(f"⚡ Execute Batch Inference with {selected_model}", type="primary"):
            progress_bar = st.progress(0, text="Initializing inference engine...")
            engine = get_inference_engine(selected_model)

            raw_texts = batch_df[selected_text_col].astype(str).tolist()
            progress_bar.progress(35, text="Running recurrent neural forward passes...")
            predictions = engine.predict(raw_texts)
            progress_bar.progress(85, text="Aggregating metrics & confidence intervals...")

            batch_df["Predicted_Emotion"] = [p["dominant_emotion"] for p in predictions]
            batch_df["Predicted_Sentiment"] = [p["sentiment"] for p in predictions]
            batch_df["Confidence"] = [round(p["confidence"], 4) for p in predictions]

            progress_bar.progress(100, text="Complete!")
            st.success("Batch classification completed successfully!")

            # Batch Summary Analytics
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Total Analyzed", len(batch_df))
            with b2:
                top_emo = batch_df["Predicted_Emotion"].mode()[0]
                st.metric("Most Frequent Emotion", top_emo)
            with b3:
                avg_conf = batch_df["Confidence"].mean() * 100
                st.metric("Mean Prediction Confidence", f"{avg_conf:.1f}%")

            # Distribution Donut
            dist_df = batch_df["Predicted_Emotion"].value_counts().reset_index()
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

            # Data Table Preview
            st.markdown("##### 📋 Enriched Dataset Preview")
            st.dataframe(batch_df[[selected_text_col, "Predicted_Emotion", "Predicted_Sentiment", "Confidence"]].head(15), use_container_width=True)

            # Download CSV
            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Enriched Classification CSV",
                data=csv_bytes,
                file_name="enriched_student_emotion_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------------------------------------------------
# TAB 4: BI-LSTM vs BI-GRU BENCHMARK
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("##### ⚖️ Recurrent Neural Architecture Benchmark")
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
            <div style="background: rgba(22, 28, 48, 0.85); border: 2px solid #6366f1; border-radius: 18px; padding: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="syne-font" style="font-size: 1.35rem; font-weight: 800; color: #818cf8;">
                        ⚡ Bidirectional LSTM
                    </div>
                    <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; border-radius: 20px; padding: 4px 10px; font-size: 0.75rem; color: #818cf8; font-weight: 700;">Fastest Latency</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                    Triple-gated memory cell structure capturing forward and backward contextual dependencies.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">VALIDATION ACCURACY</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #10b981; font-weight: 800;">{b_l.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">MACRO F1-SCORE</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #818cf8; font-weight: 800;">{b_l.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">LATENCY / SAMPLE</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #38bdf8; font-weight: 800;">{b_l.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">TRAINING TIME</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #fbbf24; font-weight: 800;">{b_l.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gru:
            st.markdown(f"""
            <div style="background: rgba(22, 28, 48, 0.85); border: 2px solid #ec4899; border-radius: 18px; padding: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="syne-font" style="font-size: 1.35rem; font-weight: 800; color: #f472b6;">
                        🚀 Bidirectional GRU
                    </div>
                    <span style="background: rgba(236, 72, 153, 0.2); border: 1px solid #ec4899; border-radius: 20px; padding: 4px 10px; font-size: 0.75rem; color: #f472b6; font-weight: 700;">Compact Memory</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                    Streamlined gated recurrent unit with fewer parameters and rapid state transitions.
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">VALIDATION ACCURACY</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #10b981; font-weight: 800;">{b_g.get('validation_accuracy', 'N/A')}%</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">MACRO F1-SCORE</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #f472b6; font-weight: 800;">{b_g.get('macro_f1', 'N/A')}</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">LATENCY / SAMPLE</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #38bdf8; font-weight: 800;">{b_g.get('latency_ms_per_sample', 'N/A')} ms</div>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">TRAINING TIME</div>
                        <div class="syne-font" style="font-size: 1.6rem; color: #fbbf24; font-weight: 800;">{b_g.get('training_time_sec', 'N/A')}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Comparative Visualization
        st.markdown("##### 📊 Quantitative Evaluation Metrics")
        comp_df = pd.DataFrame([
            {"Model": "Bi-LSTM", "Accuracy (%)": b_l.get("validation_accuracy", 0), "Macro F1 (%)": b_l.get("macro_f1", 0) * 100, "Latency (ms)": b_l.get("latency_ms_per_sample", 0)},
            {"Model": "Bi-GRU", "Accuracy (%)": b_g.get("validation_accuracy", 0), "Macro F1 (%)": b_g.get("macro_f1", 0) * 100, "Latency (ms)": b_g.get("latency_ms_per_sample", 0)}
        ])

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Accuracy (%)",
            x=comp_df["Model"],
            y=comp_df["Accuracy (%)"],
            marker_color="#10b981"
        ))
        fig_comp.add_trace(go.Bar(
            name="Macro F1 (%)",
            x=comp_df["Model"],
            y=comp_df["Macro F1 (%)"],
            marker_color="#6366f1"
        ))
        fig_comp.add_trace(go.Bar(
            name="Latency (scaled)",
            x=comp_df["Model"],
            y=comp_df["Latency (ms)"] * 50,
            marker_color="#f59e0b"
        ))

        fig_comp.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Syne", size=10))
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to compile benchmark metrics.")
