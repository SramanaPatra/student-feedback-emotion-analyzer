"""
Text Preprocessing and Sequence Tokenization utilities.
Handles vocabulary mapping, text cleaning, and sequence padding.
"""

import re
import pickle
import collections
from typing import List, Dict

# Standard emotion classes
EMOTIONS = ["Satisfaction", "Confusion", "Frustration", "Boredom", "Stress"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

class TextPreprocessor:
    def __init__(self, max_vocab_size: int = 5000, max_seq_len: int = 60):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.is_fitted = False

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower().strip()
        # Expand common contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'t", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", text)
        return text

    def fit(self, texts: List[str]):
        counter = collections.Counter()
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            counter.update(tokens)
            
        most_common = counter.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.is_fitted = True

    def transform(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]
            
            # Truncate or pad to max_seq_len
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len]
            else:
                seq = seq + [self.word2idx["<PAD>"]] * (self.max_seq_len - len(seq))
            sequences.append(seq)
        return sequences

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "max_vocab_size": self.max_vocab_size,
                "max_seq_len": self.max_seq_len,
                "is_fitted": self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls(max_vocab_size=data["max_vocab_size"], max_seq_len=data["max_seq_len"])
        obj.word2idx = data["word2idx"]
        obj.idx2word = data["idx2word"]
        obj.is_fitted = data["is_fitted"]
        return obj
