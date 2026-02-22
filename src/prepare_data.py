"""
Download and prepare email spam dataset.
2 classes: ham (0), spam (1). Emails only, no SMS.
Sources: SpamAssassin (ham+spam) + Enron (ham, subsampled).
"""

import pandas as pd
import urllib.request
import os
import io
import tarfile


def download_spamassassin():
    """SpamAssassin public corpus — real emails, ham and spam."""
    print("  📥 SpamAssassin corpus...")
    base = "https://spamassassin.apache.org/old/publiccorpus"
    archives = [
        ("20030228_easy_ham.tar.bz2", 0),
        ("20030228_easy_ham_2.tar.bz2", 0),
        ("20030228_hard_ham.tar.bz2", 0),
        ("20030228_spam.tar.bz2", 1),
        ("20030228_spam_2.tar.bz2", 1),
    ]
    rows = []
    for fname, label in archives:
        try:
            resp = urllib.request.urlopen(f"{base}/{fname}")
            with tarfile.open(fileobj=io.BytesIO(resp.read()), mode='r:bz2') as tar:
                for m in tar.getmembers():
                    if not m.isfile() or m.name.endswith('cmds'):
                        continue
                    try:
                        f = tar.extractfile(m)
                        if not f:
                            continue
                        raw = f.read().decode('utf-8', errors='ignore')
                        body = raw.split('\n\n', 1)[-1][:2000].strip()
                        if len(body) > 30:
                            rows.append({'text': body, 'label': label})
                    except Exception:
                        continue
            print(f"    ✅ {fname} loaded")
        except Exception as e:
            print(f"    ⚠️ {fname}: {e}")
    return pd.DataFrame(rows)


def download_enron():
    """Enron emails from Kaggle — legitimate corporate emails."""
    print("  📥 Enron dataset (Kaggle)...")
    import subprocess
    subprocess.run("pip install kaggle -q", shell=True, check=True)
    os.makedirs("data/raw", exist_ok=True)
    result = subprocess.run(
        "kaggle datasets download -d wcukierski/enron-email-dataset -p data/raw --unzip",
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    ⚠️ Kaggle failed: {result.stderr}")
        return pd.DataFrame()

    csv_path = next((os.path.join("data/raw", f) for f in os.listdir("data/raw") if f.endswith('.csv')), None)
    if not csv_path:
        return pd.DataFrame()

    df = pd.read_csv(csv_path, usecols=['message']).dropna()
    df['text'] = df['message'].apply(lambda m: str(m).split('\n\n', 1)[-1][:2000].strip())
    df = df[df['text'].str.len() > 30]
    df['label'] = 0
    return df[['text', 'label']]


def prepare_dataset():
    print("📊 Downloading datasets...\n")
    os.makedirs("data/processed", exist_ok=True)

    # 1. SpamAssassin — our source of both ham and spam
    sa = download_spamassassin()
    sa_ham = sa[sa['label'] == 0]
    sa_spam = sa[sa['label'] == 1]
    print(f"\n  SpamAssassin: {len(sa_ham)} ham, {len(sa_spam)} spam")

    # 2. Enron — additional ham emails
    enron = download_enron()
    print(f"  Enron: {len(enron)} ham")

    # 3. Balance the dataset
    # Target: spam count × 3 = total ham (3:1 ratio is standard for imbalanced classification)
    spam_count = len(sa_spam)
    target_ham = spam_count * 3

    # Take all SpamAssassin ham, fill rest from Enron
    enron_needed = max(0, target_ham - len(sa_ham))
    enron_sample = enron.sample(n=min(enron_needed, len(enron)), random_state=42)

    ham = pd.concat([sa_ham, enron_sample], ignore_index=True)
    if len(ham) > target_ham:
        ham = ham.sample(n=target_ham, random_state=42)

    df = pd.concat([ham, sa_spam], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n📊 Final dataset: {len(df)} samples")
    print(f"  Ham:  {(df['label'] == 0).sum()}")
    print(f"  Spam: {(df['label'] == 1).sum()}")
    print(f"  Ratio: {(df['label'] == 0).sum() / (df['label'] == 1).sum():.1f}:1")

    df.to_csv("data/processed/spam_data.csv", index=False)
    print(f"\n✅ Saved to data/processed/spam_data.csv")


if __name__ == "__main__":
    prepare_dataset()
