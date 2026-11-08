# 🎓 AI-Based Student Feedback Emotion Analyzer

An end-to-end Deep Learning system that classifies textual student feedback into granular academic emotional categories and sentiment polarity using **Bidirectional LSTM (Bi-LSTM)** and **Bidirectional GRU (Bi-GRU)** models, complete with **longitudinal trend analytics across courses and semesters**.

---

## 🚀 Key Features

1. **Dual Deep Learning Architecture (Bi-LSTM vs. Bi-GRU)**:
   - Recurrent neural networks with spatial dropout and sequence max-pooling.
   - Evaluates multi-class emotion outputs:
     - 🌟 **Satisfaction** (Positive teaching & course experience)
     - ❓ **Confusion** (Pacing, vague assignments, unclear concepts)
     - ⚠️ **Frustration** (Delayed grading, administrative / exam issues)
     - 🥱 **Boredom** (Monotonous delivery, lack of engagement)
     - ⚡ **Stress** (Overwhelming workload, tight deadlines)
   - Infers sentiment polarity: **Positive**, **Neutral**, **Negative**.

2. **Longitudinal Trend Analytics Engine (Project Extension)**:
   - **Semester-over-Semester Trajectories**: Line charts tracking shifts in satisfaction, confusion, and stress over academic terms.
   - **Net Emotion Score (NES)**: Normalized score tracking institutional sentiment:
     $$\text{NES} = \frac{\% \text{Satisfaction} - (\% \text{Confusion} + \% \text{Frustration} + \% \text{Stress})}{\text{Total Reviews}} \times 100$$
   - **Course vs Emotion Heatmaps**: Identifies which specific subjects struggle with clarity or pacing.
   - **Automated Anomaly & Spike Alerts**: Flags courses where negative emotions jumped by $\ge 10-12\%$ between terms.

3. **Interactive Web Dashboard**:
   - **Real-Time Single Feedback Tester** with preset examples and instant probability breakdown.
   - **Multi-Course & Semester Trend Visualizer** with dynamic Plotly charts.
   - **Bulk CSV Upload & Export** for processing hundreds of course reviews.
   - **Model Benchmark Tab** comparing Bi-LSTM vs. Bi-GRU in accuracy, F1-score, and latency.

---

## 📁 Project Structure

```
student_feedback_emotion_analyzer/
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Project dependencies
├── README.md                  # Complete documentation
├── data/
│   └── student_feedback.csv   # Multi-course, multi-semester feedback dataset
├── saved_models/
│   ├── bilstm_model.pt        # Trained Bi-LSTM model checkpoint
│   ├── gru_model.pt           # Trained Bi-GRU model checkpoint
│   ├── preprocessor.pkl       # Fitted tokenizer & vocabulary mapping
│   └── benchmark_results.json # Bi-LSTM vs Bi-GRU metrics comparison
└── src/
    ├── data_generator.py      # Synthetic & realistic academic dataset generator
    ├── preprocessing.py       # Text cleaning, contraction expander, sequence padding
    ├── models.py              # PyTorch Bi-LSTM and Bi-GRU architectures
    ├── train.py               # Model training & benchmarking pipeline
    ├── inference.py           # Real-time predictor engine
    └── trend_analytics.py     # Cross-course & semester trend computation engine
```

---

## 🛠️ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train / Re-train Models
```bash
python src/train.py
```
This trains both Bi-LSTM and Bi-GRU models, computes validation metrics, and updates `saved_models/`.

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
