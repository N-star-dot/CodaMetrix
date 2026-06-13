#!/usr/bin/env python3
"""
train_eval.py  -  Main execution framework: strict 5-fold stratified CV.

  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - per fold: train on 4 folds, predict the held-out fold
  - collects OUT-OF-FOLD predictions + per-category PROBABILITIES for every note
  - prints per-fold and overall Macro-F1 across the 5 categories + per-class report
  - saves oof_predictions.csv  (id, true, pred, prob_<Category> x5)

>>> PLUG YOUR MODEL IN HERE <<<  Replace `stub_model`, keep the signature:
    your_model(train_texts, train_labels, val_texts) -> (pred_labels, pred_proba)
        pred_proba columns ordered to match sorted(unique categories).
OOF probabilities are the fuel for combining two models (average/stack them).

USAGE
    python train_eval.py --medtext train.dat --mtsamples mtsamples.csv     # both (recommended)
    python train_eval.py --mtsamples mtsamples.csv                         # one source
    python train_eval.py --medtext train.dat --mtsamples mtsamples.csv --expand
"""

import argparse, collections
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from data import load_data

SEED, N_SPLITS = 42, 5


# ---- STUB MODEL: placeholder so the harness runs today. Swap in ClinicalBERT. ----
def stub_model(train_texts, train_labels, val_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=40000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(train_texts, train_labels)
    return pipe.predict(val_texts), pipe.predict_proba(val_texts)
# ----------------------------------------------------------------------------------


def run_cv(texts, labels, model_fn, n_splits=N_SPLITS):
    texts = np.array(texts, dtype=object)
    classes = np.unique(labels)
    n = len(texts)
    oof_pred = np.empty(n, dtype=object)
    oof_prob = np.zeros((n, len(classes)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(texts, labels), 1):
        preds, proba = model_fn(list(texts[tr]), labels[tr], list(texts[va]))
        oof_pred[va] = preds
        oof_prob[va, :proba.shape[1]] = proba
        s = f1_score(labels[va], preds, average="macro")
        fold_scores.append(s)
        print(f"  fold {fold}: macro-F1 = {s:.4f}  (val n={len(va)})")

    overall = f1_score(labels, oof_pred, average="macro")
    print(f"\nfold mean macro-F1 : {np.mean(fold_scores):.4f}  (+/- {np.std(fold_scores):.4f})")
    print(f"OOF   macro-F1     : {overall:.4f}\n")
    print(classification_report(labels, oof_pred, digits=3))

    out = pd.DataFrame({"id": np.arange(n), "true": labels, "pred": oof_pred})
    for j, c in enumerate(classes):
        out[f"prob_{c}"] = oof_prob[:, j]
    out.to_csv("oof_predictions.csv", index=False)
    print("saved OOF table -> oof_predictions.csv")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medtext", help="path to the 14k medical-text train.dat")
    ap.add_argument("--mtsamples", help="path to mtsamples.csv")
    ap.add_argument("--expand", action="store_true", help="apply clinical acronym expansion")
    a = ap.parse_args()

    texts, labels = load_data(medtext=a.medtext, mtsamples=a.mtsamples,
                              expand=a.expand, dedup=True)
    print(f"loaded {len(texts)} notes (dedup on, expand={a.expand})")
    print("category counts:", dict(sorted(collections.Counter(labels.tolist()).items())), "\n")
    run_cv(texts, labels, stub_model)


if __name__ == "__main__":
    main()
