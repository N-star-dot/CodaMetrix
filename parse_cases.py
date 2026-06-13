#!/usr/bin/env python3
"""
parse_cases.py  -  handle the FINAL test file. PURE DATA PIPELINE, no model.

Two jobs:
  1) Parse the final "Case N" file into clean model-ready JSONL.
  2) Build the submission CSV (case_number, prediction) once the model has
     produced predictions.

USAGE
-----
  # 1. turn the final test file into model input
  python parse_cases.py final_test.txt
        -> final_test.jsonl   (one line per case: {"id": 1, "text": "..."})

  # 2. after the model predicts (one label per line, in case order):
  python parse_cases.py final_test.txt --submission model_preds.txt
        -> submission.csv     (two columns: case_number, prediction)
"""

import sys, re, json, csv, html, unicodedata

HEADER = re.compile(r"^\s*Case\s+(\d+)\s*:?\s*$", re.IGNORECASE)


def clean_text(t):
    """Same normalization data.py uses, so the final test matches training."""
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if ch in "\n\t" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", t).strip()


def parse_cases(path):
    """Return [(case_number, text), ...] in file order. Handles multi-line cases."""
    cases, cur_id, buf = [], None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HEADER.match(line)
            if m:
                if cur_id is not None:
                    cases.append((cur_id, clean_text(" ".join(buf))))
                cur_id, buf = int(m.group(1)), []
            elif line.strip():
                buf.append(line.strip())
        if cur_id is not None:
            cases.append((cur_id, clean_text(" ".join(buf))))
    return cases


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    in_path = sys.argv[1]
    cases = parse_cases(in_path)
    print(f"parsed {len(cases)} cases")
    if cases:
        ids = [c[0] for c in cases]
        ok = ids == list(range(ids[0], ids[0] + len(ids)))
        print(f"case numbers {ids[0]}..{ids[-1]}  ({'contiguous' if ok else 'GAPS - check file'})")

    # --- mode 2: build submission CSV from model predictions ---
    if "--submission" in sys.argv:
        preds_path = sys.argv[sys.argv.index("--submission") + 1]
        with open(preds_path, encoding="utf-8", errors="replace") as f:
            preds = [ln.strip() for ln in f if ln.strip()]
        if len(preds) != len(cases):
            print(f"WARNING: {len(preds)} predictions but {len(cases)} cases — they must line up!")
        with open("submission.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_number", "prediction"])      # drop this line if no header wanted
            for (cid, _), p in zip(cases, preds):
                w.writerow([cid, p])
        print(f"wrote submission.csv ({len(preds)} rows)")
        return

    # --- mode 1 (default): write model-ready JSONL ---
    with open("final_test.jsonl", "w", encoding="utf-8") as f:
        for cid, txt in cases:
            f.write(json.dumps({"id": cid, "text": txt}, ensure_ascii=False) + "\n")
    print("wrote final_test.jsonl")


if __name__ == "__main__":
    main()
