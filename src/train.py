"""
Training script for Transformer spam detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
import time

from src.transformer_model import TransformerClassifier
from src.preprocessing import load_and_preprocess_data

CLASS_NAMES = ['ham', 'spam']


class EmailDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_model(data_path, epochs=15, batch_size=64, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), preprocessor = load_and_preprocess_data(data_path)

    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.pkl")

    train_loader = DataLoader(EmailDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmailDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(EmailDataset(X_te, y_te), batch_size=batch_size)

    config = dict(
        vocab_size=len(preprocessor.word2idx),
        d_model=256, nhead=8, num_layers=4,
        num_classes=2, max_len=256, dropout=0.1
    )
    model = TransformerClassifier(**config).to(device)
    with open("models/config.json", "w") as f:
        json.dump(config, f)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # Class weights to handle imbalance
    class_counts = np.bincount(y_tr)
    weights = torch.FloatTensor(1.0 / class_counts).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        t_loss = t_correct = t_total = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            _, pred = out.max(1)
            t_total += labels.size(0)
            t_correct += pred.eq(labels).sum().item()
        scheduler.step()

        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                out = model(seqs)
                v_loss += criterion(out, labels).item()
                _, pred = out.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()

        ta = 100. * t_correct / t_total
        va = 100. * v_correct / v_total
        history["train_loss"].append(t_loss / len(train_loader))
        history["train_acc"].append(ta)
        history["val_loss"].append(v_loss / len(val_loader))
        history["val_acc"].append(va)

        print(f"Epoch {epoch+1}/{epochs} | Train: {ta:.1f}% | Val: {va:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), "models/transformer_best.pth")
            print(f"  → Best model saved ({va:.1f}%)")

    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed/60:.1f} min")

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load("models/transformer_best.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            _, pred = model(seqs).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    metrics = {
        "history": history,
        "test_accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "training_time_min": round(elapsed / 60, 1),
        "total_params": total_params,
        "report": classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Done! Test accuracy: {metrics['test_accuracy']:.1%}")


if __name__ == "__main__":
    train_model("data/processed/spam_data.csv")
