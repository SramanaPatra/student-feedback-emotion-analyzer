"""
Streamlit Web Application: AI-Based Student Feedback Emotion Analyzer
Featuring Bi-LSTM / GRU Neural Models & Multi-Course / Semester Trend Analytics.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure imports work whether run from root or src
import sys
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
    page_title="Student Feedback Emotion Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-sat { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-neg { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-neu { background-color: #e2e3e5; color: #383d41; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Helper function to check/load default data
DATA_PATH = os.path.join(BASE_DIR, "data", "student_feedback.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

@st.cache_data
def load_default_dataset():
    if not os.path.exists(DATA_PATH):
        generate_dataset(n_samples=1800, output_csv=DATA_PATH)
    return pd.read_csv(DATA_PATH)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=110)
    st.title("Settings & Controls")
    
    selected_model = st.radio(
        "🧠 Deep Learning Architecture:",
        ["Bi-LSTM", "Bi-GRU"],
        index=0,
        help="Select between Bidirectional LSTM or Bidirectional GRU model."
    )
    
    st.markdown("---")
    st.subheader("Project Specs")
    st.info("""
    **Model:** Bi-LSTM / GRU  
    **Target Categories:**  
    • Satisfaction (Positive)  
    • Confusion (Negative)  
    • Frustration (Negative)  
    • Boredom (Neutral)  
    • Stress (Negative)  
    **Extension:** Cross-course & semester trends
    """)
    st.markdown("---")
    st.caption("AI Student Feedback Emotion Analyzer v1.0")

# App Header
st.title("🎓 AI-Based Student Feedback Emotion Analyzer")
st.markdown("Automated classification of textual feedback using **Bidirectional Recurrent Neural Networks (Bi-LSTM / GRU)** with longitudinal trend analytics across academic courses and semesters.")

# Top Navigation Tabs
tab_live, tab_trends, tab_batch, tab_benchmark = st.tabs([
    "🔍 Real-Time Emotion Tester",
    "📈 Multi-Course & Semester Trends",
    "📁 Batch CSV Processing",
    "⚖️ Bi-LSTM vs. GRU Benchmark"
])

# -------------------------------------------------------------
# TAB 1: Real-Time Feedback Tester
# -------------------------------------------------------------
with tab_live:
    st.subheader("Interactive Single-Feedback Tester")
    st.write("Enter qualitative feedback or click one of the quick test presets below:")
    
    if "input_feedback" not in st.session_state:
        st.session_state["input_feedback"] = "The professor explained complex topics with crystal clarity. Loved the hands-on labs!"

    def set_feedback_preset(text):
        st.session_state["input_feedback"] = text

    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    col_preset1.button(
        "💡 Positive / Satisfaction",
        on_click=set_feedback_preset,
        args=("The professor explained complex topics with crystal clarity. Loved the hands-on labs!",),
        use_container_width=True
    )
    col_preset2.button(
        "❓ Unclear / Confusion",
        on_click=set_feedback_preset,
        args=("Assignment instructions were vague and contradicted what was said in lecture slides.",),
        use_container_width=True
    )
    col_preset3.button(
        "⚠️ Frustration",
        on_click=set_feedback_preset,
        args=("Grading was delayed for weeks and the exam had questions never covered in class.",),
        use_container_width=True
    )
    col_preset4.button(
        "⚡ Overwhelmed / Stress",
        on_click=set_feedback_preset,
        args=("Having three major deadlines in the exact same week was completely unmanageable.",),
        use_container_width=True
    )

    input_text = st.text_area(
        "Student Feedback Comment:",
        key="input_feedback",
        height=100
    )

    if st.button("🚀 Analyze Feedback with " + selected_model, type="primary"):
        try:
            engine = EmotionInferenceEngine(model_type=selected_model, models_dir=MODELS_DIR)
            prediction = engine.predict(input_text)[0]
            
            dominant_emotion = prediction["dominant_emotion"]
            sentiment = prediction["sentiment"]
            confidence = prediction["confidence"]
            breakdown = prediction["breakdown"]

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Dominant Emotion", dominant_emotion, f"{confidence * 100:.1f}% Confidence")
            with col_res2:
                st.metric("Sentiment Polarity", sentiment)
            with col_res3:
                model_badge = "Bi-LSTM (2-Layer)" if selected_model == "Bi-LSTM" else "Bi-GRU (2-Layer)"
                st.metric("Inference Engine", model_badge)

            # Chart of probabilities
            df_probs = pd.DataFrame(list(breakdown.items()), columns=["Emotion", "Probability"])
            df_probs["Percentage"] = (df_probs["Probability"] * 100).round(1)
            
            # Color map for emotions
            color_map = {
                "Satisfaction": "#2E7D32",
                "Confusion": "#F57C00",
                "Frustration": "#C62828",
                "Boredom": "#78909C",
                "Stress": "#6A1B9A"
            }
            
            fig = px.bar(
                df_probs,
                x="Percentage",
                y="Emotion",
                orientation="h",
                color="Emotion",
                color_discrete_map=color_map,
                text="Percentage",
                title=f"Emotion Probability Distribution ({selected_model})"
            )
            fig.update_layout(showlegend=False, xaxis_title="Confidence Percentage (%)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Inference error: {str(e)}. Please verify model checkpoints exist in `{MODELS_DIR}`.")

# -------------------------------------------------------------
# TAB 2: Multi-Course & Semester Trend Analytics (Core Extension)
# -------------------------------------------------------------
with tab_trends:
    st.subheader("Longitudinal Trend Analytics Across Semesters & Courses")
    
    df_trends = load_default_dataset()
    
    # Filter Controls
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        courses_available = sorted(df_trends["course_id"].unique().tolist())
        selected_courses = st.multiselect(
            "Filter Courses:",
            options=courses_available,
            default=courses_available,
            help="Select one or more courses to compare."
        )
    with col_f2:
        semesters_available = df_trends["semester"].unique().tolist()
        st.write(f"**Semesters Monitored:** {len(semesters_available)} academic terms")

    if not selected_courses:
        st.info("ℹ️ No courses selected. Please select one or more courses from the 'Filter Courses' dropdown above to view trend analysis and heatmaps.")
    else:
        filtered_df = df_trends[df_trends["course_id"].isin(selected_courses)]

        # Summary KPI row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_reviews = len(filtered_df)
        net_score = calculate_net_emotion_score(filtered_df)
        most_common_emotion = filtered_df["emotion"].mode()[0] if len(filtered_df) > 0 else "N/A"
        satisfaction_rate = (filtered_df["emotion"] == "Satisfaction").mean() * 100 if len(filtered_df) > 0 else 0

        kpi1.metric("Total Reviews Analyzed", f"{total_reviews:,}")
        kpi2.metric("Net Emotion Score (NES)", f"{net_score:+.1f}", help="NES = %Satisfaction - %Negative Emotions")
        kpi3.metric("Dominant Overall Emotion", most_common_emotion)
        kpi4.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")

        st.markdown("---")

        # Longitudinal Trend Line Chart
        st.subheader("1. Semester-over-Semester Emotion Trajectory")
        trend_table = compute_semester_trends(filtered_df)
        
        fig_trend = go.Figure()
        emotion_colors = {
            "Satisfaction": "#2E7D32",
            "Confusion": "#F57C00",
            "Frustration": "#C62828",
            "Boredom": "#78909C",
            "Stress": "#6A1B9A"
        }
        
        for em in EMOTIONS:
            if em in trend_table.columns:
                fig_trend.add_trace(go.Scatter(
                    x=trend_table["semester"],
                    y=trend_table[em],
                    mode="lines+markers",
                    name=em,
                    line=dict(width=3, color=emotion_colors.get(em, "#333")),
                    marker=dict(size=8)
                ))
                
        fig_trend.update_layout(
            title="Emotion Trends Across Semesters (%)",
            xaxis_title="Semester",
            yaxis_title="Proportion of Feedback (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Net Emotion Score Trend
        st.subheader("2. Net Emotion Score (NES) Progression")
        fig_nes = px.bar(
            trend_table,
            x="semester",
            y="Net_Emotion_Score",
            color="Net_Emotion_Score",
            color_continuous_scale="RdYlGn",
            text_auto=".1f",
            title="Net Emotion Score Trend (-100 to +100)"
        )
        fig_nes.update_layout(yaxis_title="Net Emotion Score", coloraxis_showscale=False)
        st.plotly_chart(fig_nes, use_container_width=True)

        # Course Comparison Heatmap
        st.subheader("3. Course vs Emotion Intensity Heatmap")
        heatmap_matrix = compute_course_heatmap_matrix(filtered_df)
        
        fig_heat = px.imshow(
            heatmap_matrix,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Emotion Category", y="Course ID", color="Percentage (%)"),
            title="Course-wise Emotion Distribution (%)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Automated Spike Alerts
        st.subheader("4. Automated Pedagogical Spike Alerts")
        spikes = detect_emotion_spikes(filtered_df, threshold_percent=10.0)
        if spikes:
            for s in spikes:
                st.warning(
                    f"🚨 **Spike Alert in Course {s['course_id']}**: "
                    f"**{s['emotion']}** rose by **+{s['spike_amount']}%** in **{s['semester']}** "
                    f"(from {s['previous_rate']}% in {s['previous_semester']} to {s['current_rate']}%)."
                )
        else:
            st.success("✅ No extreme negative emotion spikes detected across current course selections.")

# -------------------------------------------------------------
# TAB 3: Batch CSV Upload
# -------------------------------------------------------------
with tab_batch:
    st.subheader("Bulk Student Feedback Evaluation")
    st.write("Upload a CSV file with student comments to automatically classify all entries.")
    
    uploaded_file = st.file_uploader("Upload Feedback CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded file contains **{len(batch_df)}** rows.")
        
        # Check text column
        text_cols = [col for col in batch_df.columns if "text" in col.lower() or "feedback" in col.lower() or "comment" in col.lower()]
        selected_text_col = st.selectbox("Select Feedback Text Column:", options=batch_df.columns, index=batch_df.columns.get_loc(text_cols[0]) if text_cols else 0)
        
        if st.button("⚡ Run Batch Classification", type="primary"):
            with st.spinner(f"Running inference with {selected_model}..."):
                engine = EmotionInferenceEngine(model_type=selected_model, models_dir=MODELS_DIR)
                predictions = engine.predict(batch_df[selected_text_col].astype(str).tolist())
                
                batch_df["Predicted_Emotion"] = [p["dominant_emotion"] for p in predictions]
                batch_df["Predicted_Sentiment"] = [p["sentiment"] for p in predictions]
                batch_df["Confidence"] = [p["confidence"] for p in predictions]
                
                st.success("Batch classification complete!")
                st.dataframe(batch_df[[selected_text_col, "Predicted_Emotion", "Predicted_Sentiment", "Confidence"]].head(10))
                
                # CSV Download button
                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Enriched Predictions CSV",
                    data=csv_data,
                    file_name="analyzed_student_feedback.csv",
                    mime="text/csv"
                )

# -------------------------------------------------------------
# TAB 4: Bi-LSTM vs. Bi-GRU Comparative Benchmark
# -------------------------------------------------------------
with tab_benchmark:
    st.subheader("Model Performance & Architecture Benchmark")
    st.write("Formal comparative assessment between Bidirectional LSTM and Bidirectional GRU:")
    
    benchmark_file = os.path.join(MODELS_DIR, "benchmark_results.json")
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)
            
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 🔹 Bidirectional LSTM")
            b_l = bench_data.get("Bi-LSTM", {})
            st.metric("Validation Accuracy", f"{b_l.get('validation_accuracy', 'N/A')}%")
            st.metric("Macro F1-Score", f"{b_l.get('macro_f1', 'N/A')}")
            st.metric("Training Time", f"{b_l.get('training_time_sec', 'N/A')} seconds")
            st.metric("Inference Latency", f"{b_l.get('latency_ms_per_sample', 'N/A')} ms/sample")
            
        with col_m2:
            st.markdown("### 🔸 Bidirectional GRU")
            b_g = bench_data.get("Bi-GRU", {})
            st.metric("Validation Accuracy", f"{b_g.get('validation_accuracy', 'N/A')}%")
            st.metric("Macro F1-Score", f"{b_g.get('macro_f1', 'N/A')}")
            st.metric("Training Time", f"{b_g.get('training_time_sec', 'N/A')} seconds")
            st.metric("Inference Latency", f"{b_g.get('latency_ms_per_sample', 'N/A')} ms/sample")
            
        st.markdown("---")
        # Visual comparison bar chart
        comp_df = pd.DataFrame([
            {"Model": "Bi-LSTM", "Accuracy (%)": b_l.get("validation_accuracy", 0), "Macro F1": b_l.get("macro_f1", 0) * 100, "Latency (ms)": b_l.get("latency_ms_per_sample", 0)},
            {"Model": "Bi-GRU", "Accuracy (%)": b_g.get("validation_accuracy", 0), "Macro F1": b_g.get("macro_f1", 0) * 100, "Latency (ms)": b_g.get("latency_ms_per_sample", 0)}
        ])
        
        fig_comp = px.bar(
            comp_df,
            x="Model",
            y=["Accuracy (%)", "Macro F1"],
            barmode="group",
            title="Bi-LSTM vs Bi-GRU Evaluation Metrics"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate the formal benchmark evaluation metrics.")
