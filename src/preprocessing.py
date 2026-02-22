"""
Data preprocessing for spam detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import pickle


class TextPreprocessor:
    """Preprocess text data for transformer model."""

    def __init__(self, vocab_size=30000, max_len=256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', ' URL ', text)
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def build_vocab(self, texts):
        all_words = []
        for text in texts:
            all_words.extend(self.clean_text(text).split())
        word_counts = Counter(all_words)
        for idx, (word, _) in enumerate(word_counts.most_common(self.vocab_size - 2), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequence(self, text):
        words = self.clean_text(text).split()
        seq = [self.word2idx.get(w, 1) for w in words]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq += [0] * (self.max_len - len(seq))
        return seq

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Split: 70/15/15 — no stratify to avoid issues with small classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['text'].values, df['label'].values,
        test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Build vocabulary
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(X_train)
    print(f"Vocabulary size: {len(preprocessor.word2idx)}")

    # Convert to sequences
    X_train_seq = np.array([preprocessor.text_to_sequence(t) for t in X_train])
    X_val_seq = np.array([preprocessor.text_to_sequence(t) for t in X_val])
    X_test_seq = np.array([preprocessor.text_to_sequence(t) for t in X_test])

    return (X_train_seq, y_train), (X_val_seq, y_val), (X_test_seq, y_test), preprocessor
