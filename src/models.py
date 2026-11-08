"""
PyTorch Deep Learning Architectures:
1. Bidirectional LSTM (Bi-LSTM) Classifier
2. Bidirectional GRU (Bi-GRU) Classifier
For multi-class student emotion and sentiment classification.
"""

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 5, num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        # Bi-directional produces hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embeds = self.dropout(self.embedding(x))
        # out shape: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (hn, cn) = self.lstm(embeds)
        
        # Max-pooling across sequence dimension to capture prominent emotion cues
        pooled, _ = torch.max(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 5, num_layers: int = 2, dropout: float = 0.3):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embeds = self.dropout(self.embedding(x))
        gru_out, hn = self.gru(embeds)
        
        # Max-pooling across sequence dimension
        pooled, _ = torch.max(gru_out, dim=1)
        logits = self.fc(pooled)
        return logits
