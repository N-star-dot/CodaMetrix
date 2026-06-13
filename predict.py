#!/usr/bin/env python3
"""
predict.py  -  Run the trained ensemble on the hidden test set.

Trains on the full dataset, then predicts the test cases and writes submission.csv.

USAGE
    python predict.py final_test.txt
    python predict.py final_test.txt --medtext train.dat --mtsamples mtsamples.csv --expand
    python predict.py final_test.txt --csv unleash_train_1k.csv
"""

import argparse, collections, subprocess, sys, os, json, tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from parse_cases import parse_cases
from data import load_data, CATEGORIES
from clinical_bert_ensemble import build_clinical_tfidf

SEED = 42


def _extract_subprocess(texts, project_dir):
    _SCRIPT = """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[3])
from clinical_bert_ensemble import extract_clinical_bert_embeddings
texts = json.load(open(sys.argv[1]))
emb = extract_clinical_bert_embeddings(texts, batch_size=32)
np.save(sys.argv[2], emb)
print(f"Extracted {emb.shape}")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(texts, f); tmp_json = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_SCRIPT); tmp_py = f.name
    tmp_npy = tmp_json.replace(".json", ".npy")
    result = subprocess.run(
        [sys.executable, tmp_py, tmp_json, tmp_npy, project_dir],
        capture_output=True, text=True
    )
    for p in [tmp_json, tmp_py]:
        os.remove(p)
    if result.returncode != 0:
        raise RuntimeError(f"BERT extraction failed:\n{result.stderr}")
    print(result.stdout.strip())
    emb = np.load(tmp_npy); os.remove(tmp_npy)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_file", help="hidden test .txt file (Case N: format)")
    ap.add_argument("--csv", default="unleash_train_1k.csv")
    ap.add_argument("--medtext")
    ap.add_argument("--mtsamples")
    ap.add_argument("--expand", action="store_true")
    a = ap.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Load training data ----
    if a.medtext or a.mtsamples:
        train_texts, train_labels = load_data(
            medtext=a.medtext, mtsamples=a.mtsamples, expand=a.expand, dedup=True)
    else:
        df = pd.read_csv(a.csv)
        train_texts = df["clinical_text"].tolist()
        train_labels = df["target_label"].values
        if a.expand:
            from data import expand_acronyms
            train_texts = [expand_acronyms(t) for t in train_texts]

    print(f"Training on {len(train_texts)} notes")
    print("category counts:", dict(sorted(collections.Counter(train_labels.tolist()).items())))

    # ---- Parse test cases ----
    cases = parse_cases(a.test_file)
    test_ids   = [c[0] for c in cases]
    test_texts = [c[1] for c in cases]
    print(f"\nTest cases: {len(test_texts)} (case {test_ids[0]}..{test_ids[-1]})")

    # ---- Extract BERT embeddings (train + test together, one subprocess) ----
    print("\nExtracting ClinicalBERT embeddings (train + test)...")
    all_texts = list(train_texts) + test_texts
    all_emb   = _extract_subprocess(all_texts, project_dir)
    train_emb = all_emb[:len(train_texts)]
    test_emb  = all_emb[len(train_texts):]
    print(f"Train emb: {train_emb.shape}  Test emb: {test_emb.shape}")

    # ---- Encode labels ----
    le = LabelEncoder(); le.fit(CATEGORIES)
    y_train = le.transform(train_labels)
    n_classes = len(CATEGORIES)
    weight_map = {"Other": 2.0, "Neurology": 1.2, "Orthopedics": 1.2}
    cw = {i: weight_map.get(c, 1.0) for i, c in enumerate(le.classes_)}

    # ---- Train on FULL dataset (no CV — use all data for final model) ----
    print("\nTraining final model on full dataset...")
    X_train_tfidf, X_test_tfidf, _ = build_clinical_tfidf(
        list(train_texts), test_texts
    )
    X_train = hstack([X_train_tfidf, csr_matrix(train_emb)]).tocsr()
    X_test  = hstack([X_test_tfidf,  csr_matrix(test_emb)]).tocsr()

    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight=cw, random_state=SEED)
    xgb_m = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=200,
        subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
        min_child_weight=3, random_state=SEED
    )
    lr.fit(X_train, y_train)
    xgb_m.fit(X_train, y_train)

    prob_lr  = lr.predict_proba(X_test)
    prob_xgb = xgb_m.predict_proba(X_test)

    # Align columns
    full_lr  = np.zeros((len(test_texts), n_classes))
    full_xgb = np.zeros((len(test_texts), n_classes))
    for ci, idx in enumerate(lr.classes_):
        full_lr[:, idx] = prob_lr[:, ci]
    for ci, idx in enumerate(xgb_m.classes_):
        full_xgb[:, idx] = prob_xgb[:, ci]

    # Default 60/40 blend (no val set to optimize against)
    blended  = 0.6 * full_lr + 0.4 * full_xgb
    preds    = le.inverse_transform(np.argmax(blended, axis=1))

    # ---- Write submission.csv ----
    out = pd.DataFrame({"case_number": test_ids, "prediction": preds})
    out.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv saved ({len(out)} rows)")
    print(out["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()
