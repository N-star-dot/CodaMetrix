#!/usr/bin/env python3
"""
train_eval.py  -  5-fold stratified CV with ClinicalBERT + TF-IDF ensemble.

USAGE
    python train_eval.py                          # uses unleash_train_1k.csv
    python train_eval.py --expand                 # + clinical acronym expansion
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


def _extract_bert_subprocess(texts, project_dir):
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


def run_cv(texts, labels_str, bert_embeddings, n_splits=N_SPLITS):
    from clinical_bert_ensemble import build_clinical_tfidf
    from scipy.sparse import hstack, csr_matrix
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    # Fix encoding to CATEGORIES order — consistent across all folds
    le = LabelEncoder()
    le.fit(CATEGORIES)
    labels = le.transform(labels_str)
    n_classes = len(CATEGORIES)

    # Build class weights — boost Other 2× and Neurology/Orthopedics 1.2×
    weight_map = {"Other": 2.0, "Neurology": 1.2, "Orthopedics": 1.2}
    cw = {i: weight_map.get(c, 1.0) for i, c in enumerate(le.classes_)}

    texts_arr = np.array(texts, dtype=object)
    n = len(texts_arr)
    oof_pred = np.empty(n, dtype=int)
    oof_prob = np.zeros((n, n_classes))

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

        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight=cw, random_state=42)
        xgb_m = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.03, n_estimators=400,
            subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
            min_child_weight=3, random_state=42, early_stopping_rounds=15
        )
        lr.fit(X_train, y_train)
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        prob_lr  = lr.predict_proba(X_val)
        prob_xgb = xgb_m.predict_proba(X_val)

        # Align proba columns to CATEGORIES order via classes_ — handles missing classes in fold
        full_lr  = np.zeros((len(va), n_classes))
        full_xgb = np.zeros((len(va), n_classes))
        for ci, idx in enumerate(lr.classes_):
            full_lr[:, idx] = prob_lr[:, ci]
        for ci, idx in enumerate(xgb_m.classes_):
            full_xgb[:, idx] = prob_xgb[:, ci]

        best_f1, best_w = 0, 0.5
        for w in np.linspace(0.1, 0.9, 9):
            preds = np.argmax(w * full_lr + (1-w) * full_xgb, axis=1)
            score = f1_score(y_val, preds, average="macro")
            if score > best_f1:
                best_f1, best_w = score, w
        print(f"  Best blend weight (LR): {best_w:.2f}")

        blended = best_w * full_lr + (1 - best_w) * full_xgb
        oof_prob[va] = blended
        oof_pred[va] = np.argmax(blended, axis=1)

        s = f1_score(y_val, oof_pred[va], average="macro")
        fold_scores.append(s)
        print(f"  fold {fold}: macro-F1 = {s:.4f}  (val n={len(va)})")

    overall = f1_score(labels, oof_pred, average="macro")
    print(f"\nfold mean macro-F1 : {np.mean(fold_scores):.4f}  (+/- {np.std(fold_scores):.4f})")
    print(f"OOF   macro-F1     : {overall:.4f}\n")
    print(classification_report(labels, oof_pred,
                                labels=list(range(n_classes)),
                                target_names=le.classes_, digits=3))

    out = pd.DataFrame({"id": np.arange(n), "true": labels_str,
                        "pred": le.inverse_transform(oof_pred)})
    for j, c in enumerate(le.classes_):
        out[f"prob_{c}"] = oof_prob[:, j]
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

    print(f"loaded {len(texts)} notes")
    print("category counts:", dict(sorted(collections.Counter(labels_str.tolist()).items())), "\n")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("Extracting ClinicalBERT embeddings (MPS subprocess)...")
    bert_emb = _extract_bert_subprocess(list(texts), project_dir)
    print(f"Embeddings shape: {bert_emb.shape}\n")

    run_cv(texts, labels_str, bert_emb)


if __name__ == "__main__":
    main()
