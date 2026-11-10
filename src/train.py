"""
Training and Evaluation Pipeline for Bi-LSTM and Bi-GRU Models.
Compares both architectures on the enriched student feedback corpus and saves checkpoints.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import copy
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.preprocessing import TextPreprocessor, EMOTIONS
from src.models import BiLSTMClassifier, BiGRUClassifier
from src.data_generator import generate_dataset

def train_one_model(model_cls, name, X_train, y_train, X_val, y_val, vocab_size, epochs=12, batch_size=32):
    print(f"\n--- Training {name} ---")
    model = model_cls(vocab_size=vocab_size, embed_dim=64, hidden_dim=64, num_classes=len(EMOTIONS), dropout=0.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    start_time = time.time()
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        epoch_preds = []
        epoch_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).numpy()
                epoch_preds.extend(preds)
                epoch_targets.extend(batch_y.numpy())

        epoch_acc = accuracy_score(epoch_targets, epoch_preds)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_weights = copy.deepcopy(model.state_dict())

    training_duration = round(time.time() - start_time, 2)
    
    # Restore best weights and perform final evaluation
    model.load_state_dict(best_weights)
    model.eval()
    all_preds = []
    all_targets = []
    
    infer_start = time.time()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
    infer_latency = round((time.time() - infer_start) * 1000 / len(all_targets), 3) # ms per sample
    
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    
    print(f"{name} Completed: Train Time={training_duration}s | Best Val Acc={acc*100:.2f}% | Macro F1={macro_f1:.4f} | Latency={infer_latency}ms")
    
    return model, {
        "model_name": name,
        "training_time_sec": training_duration,
        "validation_accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "latency_ms_per_sample": infer_latency
    }


def run_training_pipeline(data_path="data/student_feedback.csv", output_dir="saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ensure dataset exists
    if not os.path.exists(data_path):
        print("Dataset not found. Generating academic dataset...")
        df = generate_dataset(n_samples=3600, output_csv=data_path)
    else:
        df = pd.read_csv(data_path)
        
    print(f"Loaded {len(df)} feedback records from {data_path}.")
    
    # 2. Fit Preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=5000, max_seq_len=60)
    preprocessor.fit(df["feedback_text"].tolist())
    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))
    print(f"Vocabulary fitted with {len(preprocessor.word2idx)} unique tokens.")
    
    # 3. Prepare Tensors
    X = np.array(preprocessor.transform(df["feedback_text"].tolist()))
    label_map = {name: idx for idx, name in enumerate(EMOTIONS)}
    y = np.array([label_map[e] for e in df["emotion"].tolist()])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Bi-LSTM
    bilstm_model, bilstm_metrics = train_one_model(
        BiLSTMClassifier, "Bi-LSTM", X_train, y_train, X_val, y_val, len(preprocessor.word2idx), epochs=12
    )
    torch.save(bilstm_model.state_dict(), os.path.join(output_dir, "bilstm_model.pt"))
    
    # 5. Train Bi-GRU
    gru_model, gru_metrics = train_one_model(
        BiGRUClassifier, "Bi-GRU", X_train, y_train, X_val, y_val, len(preprocessor.word2idx), epochs=12
    )
    torch.save(gru_model.state_dict(), os.path.join(output_dir, "gru_model.pt"))
    
    # 6. Save Benchmark Results
    benchmarks = {
        "Bi-LSTM": bilstm_metrics,
        "Bi-GRU": gru_metrics,
        "emotions": EMOTIONS
    }
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(benchmarks, f, indent=4)
        
    print(f"\nAll models and preprocessors saved to {output_dir}/")
    return benchmarks

if __name__ == "__main__":
    run_training_pipeline()
