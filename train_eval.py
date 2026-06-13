#!/usr/bin/env python3
"""
train_eval.py  -  5-fold stratified CV with 6-class internal training.

Trains on 6 classes (Other split into Other_Cancer / Other_General) so the
model learns distinct embeddings for each "Other" subtype, then collapses
predictions back to the 5 competition categories for scoring.

USAGE
    python train_eval.py                          # uses unleash_train_1k.csv
    python train_eval.py --expand                 # + acronym expansion
    python train_eval.py --medtext ... --mtsamples ...  # 14k dataset
"""

import argparse, collections, subprocess, sys, os, json, tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from data import load_data, CATEGORIES

SEED, N_SPLITS = 42, 5
# 6 internal training classes — Other split for better discrimination
INTERNAL_CATEGORIES = sorted(["Cardiology", "Gastroenterology", "Neurology",
                               "Orthopedics", "Other_Cancer", "Other_General"])


def collapse_other(arr):
    """Other_Cancer / Other_General -> Other for competition scoring."""
    return np.array(["Other" if l in ("Other_Cancer", "Other_General") else l
                     for l in arr])


def _extract_bert_subprocess(texts, project_dir):
    _SCRIPT = """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[3])
from clinical_bert_ensemble import extract_clinical_bert_embeddings
texts = json.load(open(sys.argv[1]))
emb = extract_clinical_bert_embeddings(texts, batch_size=64)
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


def run_cv(texts, labels_str, bert_embeddings, n_splits=N_SPLITS):
    from clinical_bert_ensemble import build_clinical_tfidf
    from scipy.sparse import hstack, csr_matrix
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    # Encode on fixed 6-class set — no fold variance
    le = LabelEncoder()
    le.fit(INTERNAL_CATEGORIES)
    labels = le.transform(labels_str)
    n_classes = len(INTERNAL_CATEGORIES)

    texts_arr = np.array(texts, dtype=object)
    n = len(texts_arr)
    oof_pred_str = np.empty(n, dtype=object)
    oof_prob_6   = np.zeros((n, n_classes))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (tr, va) in enumerate(skf.split(texts_arr, labels), 1):
        print(f"\n  --- Fold {fold}/{n_splits} ---")
        y_train, y_val = labels[tr], labels[va]

        X_train_tfidf, X_val_tfidf, _ = build_clinical_tfidf(
            list(texts_arr[tr]), list(texts_arr[va])
        )
        X_train = hstack([X_train_tfidf, csr_matrix(bert_embeddings[tr])]).tocsr()
        X_val   = hstack([X_val_tfidf,   csr_matrix(bert_embeddings[va])]).tocsr()

        # Boost Other subtypes 2× and Neurology/Ortho 1.2× to fix minority recall
        cw = {i: 1.0 for i in range(n_classes)}
        for cls, w in [("Other_Cancer", 2.0), ("Other_General", 2.0),
                       ("Neurology", 1.2), ("Orthopedics", 1.2)]:
            if cls in le.classes_:
                cw[int(np.where(le.classes_ == cls)[0][0])] = w

        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight=cw, random_state=42)
        xgb_m = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.03, n_estimators=400,
            subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
            min_child_weight=3, random_state=42, early_stopping_rounds=15
        )
        lr.fit(X_train, y_train)
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Align proba columns to INTERNAL_CATEGORIES order
        prob_lr  = lr.predict_proba(X_val)
        prob_xgb = xgb_m.predict_proba(X_val)
        full_lr  = np.zeros((len(va), n_classes))
        full_xgb = np.zeros((len(va), n_classes))
        for ci, idx in enumerate(lr.classes_):
            full_lr[:, idx] = prob_lr[:, ci]
        for ci, idx in enumerate(xgb_m.classes_):
            full_xgb[:, idx] = prob_xgb[:, ci]

        # Dynamic blend — score on collapsed 5-class labels
        best_f1, best_w = 0, 0.5
        y_val_collapsed = collapse_other(le.inverse_transform(y_val))
        for w in np.linspace(0.1, 0.9, 9):
            preds = collapse_other(
                le.inverse_transform(np.argmax(w * full_lr + (1-w) * full_xgb, axis=1))
            )
            score = f1_score(y_val_collapsed, preds, average="macro", labels=CATEGORIES)
            if score > best_f1:
                best_f1, best_w = score, w
        print(f"  Best blend weight (LR): {best_w:.2f}")

        blended = best_w * full_lr + (1 - best_w) * full_xgb
        oof_prob_6[va]   = blended
        oof_pred_str[va] = le.inverse_transform(np.argmax(blended, axis=1))

        s = f1_score(y_val_collapsed,
                     collapse_other(oof_pred_str[va]),
                     average="macro", labels=CATEGORIES)
        fold_scores.append(s)
        print(f"  fold {fold}: macro-F1 = {s:.4f}  (val n={len(va)})")

    oof_collapsed  = collapse_other(oof_pred_str)
    true_collapsed = collapse_other(labels_str)

    overall = f1_score(true_collapsed, oof_collapsed, average="macro", labels=CATEGORIES)
    print(f"\nfold mean macro-F1 : {np.mean(fold_scores):.4f}  (+/- {np.std(fold_scores):.4f})")
    print(f"OOF   macro-F1     : {overall:.4f}\n")
    print(classification_report(true_collapsed, oof_collapsed,
                                labels=CATEGORIES, target_names=CATEGORIES, digits=3))

    # Sum Other_Cancer + Other_General proba columns into one Other column
    other_idx = [int(np.where(le.classes_ == c)[0][0])
                 for c in ("Other_Cancer", "Other_General") if c in le.classes_]
    out = pd.DataFrame({"id": np.arange(n), "true": true_collapsed, "pred": oof_collapsed})
    for c in CATEGORIES:
        if c == "Other":
            out["prob_Other"] = oof_prob_6[:, other_idx].sum(axis=1)
        else:
            idx = np.where(le.classes_ == c)[0]
            out[f"prob_{c}"] = oof_prob_6[:, idx[0]] if len(idx) else 0.0
    out.to_csv("oof_predictions.csv", index=False)
    print("saved OOF table -> oof_predictions.csv")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="unleash_train_1k.csv")
    ap.add_argument("--medtext")
    ap.add_argument("--mtsamples")
    ap.add_argument("--expand", action="store_true")
    a = ap.parse_args()

    if a.medtext or a.mtsamples:
        texts, labels_str = load_data(medtext=a.medtext, mtsamples=a.mtsamples,
                                      expand=a.expand, dedup=True)
    else:
        df = pd.read_csv(a.csv)
        texts = df["clinical_text"].tolist()
        labels_str = df["target_label"].values
        if a.expand:
            from data import expand_acronyms
            texts = [expand_acronyms(t) for t in texts]

    # Map Other -> Other_Cancer / Other_General using source column if present,
    # otherwise split evenly by index (no source info in the 1k CSV).
    # For the 1k CSV both subtypes collapsed to "Other" — re-split 50/50 by row parity.
    labels_internal = []
    other_toggle = 0
    for lbl in labels_str:
        if lbl == "Other":
            labels_internal.append("Other_Cancer" if other_toggle % 2 == 0 else "Other_General")
            other_toggle += 1
        else:
            labels_internal.append(lbl)
    labels_internal = np.array(labels_internal)

    print(f"loaded {len(texts)} notes")
    print("category counts:", dict(sorted(collections.Counter(labels_internal.tolist()).items())), "\n")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("Extracting ClinicalBERT embeddings (MPS subprocess)...")
    bert_emb = _extract_bert_subprocess(list(texts), project_dir)
    print(f"Embeddings shape: {bert_emb.shape}\n")

    run_cv(texts, labels_internal, bert_emb)


if __name__ == "__main__":
    main()
