"""
Trend Analysis and Longitudinal Analytics Engine.
Calculates emotion distributions, semester-over-semester shifts,
Net Emotion Score (NES), and anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

SEMESTER_ORDER = [
    "Spring 2024",
    "Fall 2024",
    "Spring 2025",
    "Fall 2025",
    "Spring 2026"
]

def ensure_semester_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts dataframe chronologically by semester."""
    df_copy = df.copy()
    if "semester" in df_copy.columns:
        df_copy["semester"] = pd.Categorical(df_copy["semester"], categories=SEMESTER_ORDER, ordered=True)
        return df_copy.sort_values("semester")
    return df_copy

def calculate_net_emotion_score(df: pd.DataFrame) -> float:
    """
    Computes Net Emotion Score (NES):
    NES = ((Positive / Satisfaction) - (Negative / Confusion + Frustration + Stress)) / Total * 100
    Range: -100 to +100
    """
    if len(df) == 0:
        return 0.0
    satisfaction_count = (df["emotion"] == "Satisfaction").sum()
    negatives = df["emotion"].isin(["Confusion", "Frustration", "Stress"]).sum()
    return round(((satisfaction_count - negatives) / len(df)) * 100, 2)

def compute_semester_trends(df: pd.DataFrame, course_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes the percentage of each emotion across semesters.
    """
    data = df.copy()
    if course_filter:
        data = data[data["course_id"].isin(course_filter)]
        
    data = ensure_semester_order(data)
    
    # Cross-tabulate Semester vs Emotion
    grouped = pd.crosstab(data["semester"], data["emotion"], normalize="index") * 100
    grouped = grouped.reset_index()
    
    # Add Net Emotion Score column per semester
    nes_per_sem = []
    for sem in grouped["semester"]:
        sem_df = data[data["semester"] == sem]
        nes_per_sem.append(calculate_net_emotion_score(sem_df))
    grouped["Net_Emotion_Score"] = nes_per_sem
    return grouped

def compute_course_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns Course vs Emotion percentage matrix for heatmaps.
    """
    matrix = pd.crosstab(df["course_id"], df["emotion"], normalize="index") * 100
    return matrix.round(1)

def detect_emotion_spikes(df: pd.DataFrame, threshold_percent: float = 12.0) -> List[Dict]:
    """
    Detects sudden spikes in negative emotions (Confusion, Stress, Frustration)
    between consecutive semesters for each course.
    """
    alerts = []
    courses = df["course_id"].unique()
    
    for c in courses:
        c_df = df[df["course_id"] == c]
        c_df = ensure_semester_order(c_df)
        
        tab = pd.crosstab(c_df["semester"], c_df["emotion"], normalize="index") * 100
        
        for emotion in ["Confusion", "Frustration", "Stress"]:
            if emotion in tab.columns:
                series = tab[emotion]
                for i in range(1, len(series)):
                    prev_val = series.iloc[i - 1]
                    curr_val = series.iloc[i]
                    diff = curr_val - prev_val
                    if diff >= threshold_percent:
                        alerts.append({
                            "course_id": str(c),
                            "semester": str(series.index[i]),
                            "previous_semester": str(series.index[i - 1]),
                            "emotion": str(emotion),
                            "spike_amount": round(float(diff), 1),
                            "current_rate": round(float(curr_val), 1),
                            "previous_rate": round(float(prev_val), 1)
                        })
    return alerts
