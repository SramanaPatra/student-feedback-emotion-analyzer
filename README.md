# 🎓 AI-Based Student Feedback Emotion Analyzer

An end-to-end Deep Learning system that classifies textual student feedback into granular academic emotional categories and sentiment polarity using **Bidirectional LSTM (Bi-LSTM)** and **Bidirectional GRU (Bi-GRU)** models, complete with **longitudinal trend analytics across courses and semesters** and **actionable pedagogical intervention directives**.

---

## 🚀 Key Features

### 1. Dual Deep Learning Architecture (Bi-LSTM vs. Bi-GRU)
- Recurrent neural networks with spatial dropout, bidirectional hidden representations, and sequence max-pooling.
- Multi-class affective classification across 5 granular categories:
  - 🌟 **Satisfaction** (Positive teaching & course experience, conceptual clarity)
  - ❓ **Confusion** (Pacing issues, vague assignments, contradictory instructions)
  - ⚠️ **Frustration** (Delayed grading, administrative / exam issues, unresponsive support)
  - 🥱 **Boredom** (Monotonous delivery, lack of interactive engagement)
  - ⚡ **Stress** (Overwhelming workload, tight clustered deadlines, burnout)
- Sentiment polarity inference: **Positive**, **Neutral**, **Negative**.

### 2. Longitudinal Trend Analytics Engine (Project Extension)
- **Semester-over-Semester Trajectories**: Line charts tracking shifts in satisfaction, confusion, and stress over academic terms.
- **Net Emotion Score (NES)**: Normalized score tracking institutional sentiment:
  $$\text{NES} = \frac{\% \text{Satisfaction} - (\% \text{Confusion} + \% \text{Frustration} + \% \text{Stress})}{\text{Total Reviews}} \times 100$$
- **Course vs Emotion Heatmaps**: Identifies which specific subjects struggle with clarity or pacing.
- **Automated Anomaly & Spike Alerts**: Flags courses where negative emotions jumped by $\ge 10\%$ between terms.

### 3. Actionable Pedagogical Intervention Directives
- Automated recommendations tailored to identified student sentiment bottlenecks:
  - **Satisfaction**: Archive successful lab formats & active learning setups as departmental benchmarks.
  - **Confusion**: Publish structured rubrics with code examples and host mid-term clarification Q&As.
  - **Frustration**: Audit TA response cycles, streamline auto-graders, and review exam questions against the syllabus.
  - **Boredom**: Introduce active-learning segments, live coding sessions, and peer problem-solving.
  - **Stress**: Coordinate across department instructors to de-conflict deadline clusters.

### 4. Interactive & Expressive Web Dashboard
- **Real-Time Single Feedback Tester** with preset examples and sub-millisecond cached inference.
- **5-Axis Affective Radar Chart** (`go.Scatterpolar`) mapping emotional intensity.
- **Sentiment Valence Gauge Dial** (-1.0 to +1.0) with dynamic physical-style feedback.
- **Multi-Course & Semester Trend Visualizer** with dynamic Plotly charts.
- **Bulk CSV Upload & Export** for processing hundreds of course reviews with mini-batched chunking.
- **Model Benchmark Tab** comparing Bi-LSTM vs. Bi-GRU in accuracy, F1-score, and latency.

---

## 📁 Project Structure

```
student_feedback_emotion_analyzer/
├── app.py                     # Expressive Streamlit Web Application
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── test_student_comments.csv  # 50-comment sample evaluation dataset
├── .streamlit/
│   └── config.toml            # Persistent dark theme configuration
├── data/
│   └── student_feedback.csv   # 3,000-sample multi-course, multi-semester dataset
├── saved_models/
│   ├── bilstm_model.pt        # Trained Bi-LSTM model checkpoint (100% accuracy)
│   ├── gru_model.pt           # Trained Bi-GRU model checkpoint (100% accuracy)
│   ├── preprocessor.pkl       # Fitted tokenizer & vocabulary mapping
│   └── benchmark_results.json # Bi-LSTM vs Bi-GRU metrics comparison
└── src/
    ├── data_generator.py      # Coherent academic dataset generator
    ├── preprocessing.py       # Text cleaning, contraction expander, sequence padding
    ├── models.py              # PyTorch Bi-LSTM and Bi-GRU architectures
    ├── train.py               # Model training & benchmarking pipeline
    ├── inference.py           # Real-time predictor engine with mini-batch chunking
    └── trend_analytics.py     # Cross-course & semester trend computation engine
```

---

## 🛠️ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train Models
```bash
python src/train.py
```
This trains both Bi-LSTM and Bi-GRU models on the 3,000-sample dataset, computes validation metrics, and updates `saved_models/`.

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
