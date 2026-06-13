"""
train.py — Run the full pipeline on the full dataset (14k+ notes).
BERT extraction runs in a subprocess to avoid MPS/XGBoost conflict on Apple Silicon.
"""
import subprocess
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import kagglehub
from data import load_data

# =============================================================================
# Load & encode
# =============================================================================
print("Downloading datasets via kagglehub...")
path_1 = kagglehub.dataset_download("chaitanyakck/medical-text")
path_2 = kagglehub.dataset_download("tboyle10/medicaltranscriptions")

medtext_path = os.path.join(path_1, "train.dat")
mtsamples_path = os.path.join(path_2, "mtsamples.csv")

print("Loading and preparing data...")
texts, labels_raw = load_data(medtext=medtext_path, mtsamples=mtsamples_path, expand=True)

le = LabelEncoder()
labels = le.fit_transform(labels_raw)
class_names = list(le.classes_)

print(f"Classes: {class_names}")
print(f"Label mapping: { {c: i for i, c in enumerate(class_names)} }")
print(f"Dataset shape: ({len(texts)}, 2)\n")

# =============================================================================
# BERT extraction in isolated subprocess (avoids MPS/XGBoost segfault)
# =============================================================================
_EXTRACT_SCRIPT = """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[3])
from clinical_bert_ensemble import extract_clinical_bert_embeddings

texts = json.load(open(sys.argv[1]))
emb = extract_clinical_bert_embeddings(texts, batch_size=16)
np.save(sys.argv[2], emb)
print(f"Saved embeddings {emb.shape}")
"""

print("[1/2] Extracting ClinicalBERT embeddings via subprocess (MPS)...")
with open("/tmp/train_texts.json", "w") as f:
    json.dump(texts, f)
with open("/tmp/_train_extractor.py", "w") as f:
    f.write(_EXTRACT_SCRIPT)

result = subprocess.run(
    [sys.executable, "/tmp/_train_extractor.py",
     "/tmp/train_texts.json",
     "/tmp/train_bert_emb.npy",
     os.path.dirname(os.path.abspath(__file__))],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("EXTRACTION FAILED:\n", result.stderr)
    sys.exit(1)
print(result.stdout.strip())

bert_embeddings = np.load("/tmp/train_bert_emb.npy")
print(f"Embeddings shape: {bert_embeddings.shape}\n")

# =============================================================================
# Run pipeline (XGBoost runs cleanly — no MPS in this process)
# =============================================================================
print("[2/2] Running 5-Fold CV + Dynamic Blend + Threshold Calibration...")
from clinical_bert_ensemble import run_pipeline

oof_preds, _ = run_pipeline(
    raw_texts=texts,
    bert_embeddings=bert_embeddings,
    labels=labels,
    n_classes=len(class_names),
    class_names=class_names
)

# Cleanup
for f in ["/tmp/train_texts.json", "/tmp/_train_extractor.py", "/tmp/train_bert_emb.npy"]:
    os.remove(f)
