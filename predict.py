#!/usr/bin/env python3
"""
predict.py  -  End-to-end inference for the final "Case N" test file.

Uses HIS pipeline end to end:
    training data  = his balanced unleash_train_1k.csv (200/class, correct labels,
                     case+punctuation preserved, acronyms already expanded, deduped)
    model          = clinical_bert_ensemble.run_pipeline (ClinicalBERT + TF-IDF fused,
                     LR/XGB soft-vote blend, threshold calibration) — his methods
    test cleaning  = parse_cases() + acronym expansion, identical to the training data
    -> writes submission.csv (case_number, prediction)

USAGE
    python predict.py final_test.txt
    python predict.py final_test.txt --csv unleash_train_1k.csv --out submission.csv
"""

import argparse, csv, collections, os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data import expand_acronyms, truncate_note, CATEGORIES
from parse_cases import parse_cases
from train_eval import _extract_bert_subprocess
from clinical_bert_ensemble import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_file", help="the 'Case N' final test file")
    ap.add_argument("--csv", default="unleash_train_1k.csv",
                    help="his balanced training CSV (default: unleash_train_1k.csv)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--other-prior", type=float, default=2.7,
                    help="test-time multiplier on the Other class weight (2.7 = peak acc vs ChatGPT labels)")
    a = ap.parse_args()
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. training data = his path. CSV is already cleaned + acronym-expanded.
    df = pd.read_csv(a.csv)
    # Truncate training notes to the test-vignette domain (first ~3 sentences / ~500 chars)
    # so note-length stops being a class proxy. See data.truncate_note.
    texts = [truncate_note(t) for t in df["clinical_text"].tolist()]
    labels_str = df["target_label"].values
    print(f"trained on {len(texts)} notes from {a.csv} "
          f"{dict(sorted(collections.Counter(labels_str.tolist()).items()))}")

    le = LabelEncoder().fit(CATEGORIES)
    labels = le.transform(labels_str)
    class_names = list(le.classes_)

    # 2. test cases — parse_cases cleans (keeps case+punct); expand acronyms to match training
    cases = parse_cases(a.test_file)
    if not cases:
        raise SystemExit("no cases parsed — check the test file format ('Case N' headers)")
    ids = [cid for cid, _ in cases]
    # Same truncation as training (vignettes are already short, so this is mostly a no-op
    # for them, but keeps train/test preprocessing identical).
    test_texts = [truncate_note(expand_acronyms(txt)) for _, txt in cases]
    print(f"parsed {len(cases)} cases (ids {ids[0]}..{ids[-1]})")

    # 3. BERT features for train + test (isolated subprocess: avoids MPS/XGBoost segfault)
    print("Extracting ClinicalBERT embeddings for TRAIN...")
    train_bert = _extract_bert_subprocess(texts, project_dir)
    print("Extracting ClinicalBERT embeddings for TEST...")
    test_bert = _extract_bert_subprocess(test_texts, project_dir)

    # 4. his ensemble; test_preds come back label-encoded
    _, test_preds = run_pipeline(
        raw_texts=texts,
        bert_embeddings=train_bert,
        labels=labels,
        test_texts=test_texts,
        test_bert=test_bert,
        n_classes=len(CATEGORIES),
        class_names=class_names,
        other_prior=a.other_prior,
    )
    preds = le.inverse_transform(test_preds)      # int -> category string

    # 5. submission
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_number", "prediction"])
        for cid, p in zip(ids, preds):
            w.writerow([cid, p])
    print(f"wrote {a.out} ({len(preds)} rows)")
    print("prediction counts:", dict(sorted(collections.Counter(preds.tolist()).items())))


if __name__ == "__main__":
    main()
