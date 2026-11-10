"""
Inference Engine for Web Application.
Performs single-text or batch inference using saved Bi-LSTM / GRU models.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union

from src.preprocessing import TextPreprocessor, EMOTIONS
from src.models import BiLSTMClassifier, BiGRUClassifier

EMOTION_TO_SENTIMENT = {
    "Satisfaction": "Positive",
    "Confusion": "Negative",
    "Frustration": "Negative",
    "Boredom": "Neutral",
    "Stress": "Negative"
}

class EmotionInferenceEngine:
    def __init__(self, model_type: str = "Bi-LSTM", models_dir: str = "saved_models"):
        self.models_dir = models_dir
        self.model_type = model_type
        
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Please run train.py first.")
            
        self.preprocessor = TextPreprocessor.load(preprocessor_path)
        vocab_size = len(self.preprocessor.word2idx)
        
        if model_type == "Bi-LSTM":
            self.model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, num_classes=len(EMOTIONS))
            weight_path = os.path.join(models_dir, "bilstm_model.pt")
        else:
            self.model = BiGRUClassifier(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, num_classes=len(EMOTIONS))
            weight_path = os.path.join(models_dir, "gru_model.pt")
            
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
        else:
            raise FileNotFoundError(f"Weights not found at {weight_path}")
            
        self.model.eval()

    def predict(self, texts: Union[str, List[str]], batch_size: int = 256) -> List[Dict]:
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            
        seqs = self.preprocessor.transform(texts)
        tensor_x = torch.tensor(seqs, dtype=torch.long)
        
        with torch.no_grad():
            if len(tensor_x) <= batch_size:
                logits = self.model(tensor_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            else:
                probs_list = []
                for i in range(0, len(tensor_x), batch_size):
                    chunk = tensor_x[i:i + batch_size]
                    logits = self.model(chunk)
                    chunk_probs = F.softmax(logits, dim=1).cpu().numpy()
                    probs_list.append(chunk_probs)
                probs = np.vstack(probs_list)
            
        results = []
        for prob in probs:
            idx = int(np.argmax(prob))
            dominant_emotion = EMOTIONS[idx]
            sentiment = EMOTION_TO_SENTIMENT[dominant_emotion]
            
            breakdown = {EMOTIONS[i]: round(float(prob[i]), 4) for i in range(len(EMOTIONS))}
            results.append({
                "dominant_emotion": dominant_emotion,
                "sentiment": sentiment,
                "confidence": round(float(prob[idx]), 4),
                "breakdown": breakdown
            })
            
        return results
