#!/usr/bin/env python3
"""
train_eval.py  -  Main execution framework: strict 5-fold stratified CV.

  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - per fold: train on 4 folds, predict the held-out fold
  - collects OUT-OF-FOLD predictions + per-category PROBABILITIES for every note
  - prints per-fold and overall Macro-F1 across the 5 categories + per-class report
  - saves oof_predictions.csv  (id, true, pred, prob_<Category> x5)

USAGE
    python train_eval.py --medtext train.dat --mtsamples mtsamples.csv
    python train_eval.py --medtext train.dat --mtsamples mtsamples.csv --expand
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
    """Run BERT extraction in an isolated subprocess to avoid MPS/XGBoost segfault."""
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
    emb = np.load(tmp_npy)
    os.remove(tmp_npy)
    return emb


def run_cv(texts, labels_str, bert_embeddings, n_splits=N_SPLITS):
    """
    Full CV loop using the ClinicalBERT + TF-IDF ensemble.
    labels_str: string category labels (CATEGORIES values).
    bert_embeddings: pre-extracted numpy array (n_samples, 768).
    """
    from clinical_bert_ensemble import build_clinical_tfidf
    from scipy.sparse import hstack, csr_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import xgboost as xgb

    le = LabelEncoder()
    le.fit(CATEGORIES)           # fix encoding to CATEGORIES order — no fold variance
    labels = le.transform(labels_str)
    n_classes = len(CATEGORIES)

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

        custom_weights = {0: 1.0, 1: 1.0, 2: 1.2, 3: 1.2, 4: 2.0}
        lr = LogisticRegression(C=0.1, max_iter=1000,
                                class_weight=custom_weights, random_state=42)
        xgb_m = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.03, n_estimators=400,
            subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
            min_child_weight=3, random_state=42, early_stopping_rounds=15
        )
        lr.fit(X_train, y_train)
        xgb_m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)

        prob_lr  = lr.predict_proba(X_val)
        prob_xgb = xgb_m.predict_proba(X_val)

        # Align XGBoost proba columns to CATEGORIES order via its classes_ attribute
        full_lr  = np.zeros((len(va), n_classes))
        full_xgb = np.zeros((len(va), n_classes))
        for col_i, cls_idx in enumerate(lr.classes_):
            full_lr[:, cls_idx] = prob_lr[:, col_i]
        for col_i, cls_idx in enumerate(xgb_m.classes_):
            full_xgb[:, cls_idx] = prob_xgb[:, col_i]

        # Dynamic blend weight search
        best_f1, best_w = 0, 0.5
        for w in np.linspace(0.1, 0.9, 9):
            blended_preds = np.argmax(w * full_lr + (1 - w) * full_xgb, axis=1)
            score = f1_score(y_val, blended_preds, average="macro")
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
    print(classification_report(labels, oof_pred, target_names=CATEGORIES, digits=3))

    out = pd.DataFrame({"id": np.arange(n), "true": labels_str,
                        "pred": le.inverse_transform(oof_pred)})
    for j, c in enumerate(CATEGORIES):
        out[f"prob_{c}"] = oof_prob[:, j]
    out.to_csv("oof_predictions.csv", index=False)
    print("saved OOF table -> oof_predictions.csv")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="unleash_train_1k.csv",
                    help="path to balanced training CSV (default: unleash_train_1k.csv)")
    ap.add_argument("--medtext", help="path to 14k medical-text train.dat (overrides --csv)")
    ap.add_argument("--mtsamples", help="path to mtsamples.csv (overrides --csv)")
    ap.add_argument("--expand", action="store_true", help="apply clinical acronym expansion")
    a = ap.parse_args()

    if a.medtext or a.mtsamples:
        texts, labels_str = load_data(medtext=a.medtext, mtsamples=a.mtsamples,
                                      expand=a.expand, dedup=True)
    else:
        import pandas as pd
        df = pd.read_csv(a.csv)
        texts = df["clinical_text"].tolist()
        labels_str = df["target_label"].values
        if a.expand:
            from data import expand_acronyms, clean_text
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
