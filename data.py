#!/usr/bin/env python3
"""
data.py  -  Data layer for the 5-category clinical classifier.

TARGET CATEGORIES (what you train + submit on):
    Cardiology, Neurology, Orthopedics, Gastroenterology, Other

TWO SOURCES, both mapped into those 5 buckets. Use either or both:
  * medical-text 14k abstracts (label\\ttext, labels 1-5)   -> load via medtext=
        good VOLUME + matches the final test's abstract style, BUT has NO Orthopedics.
  * MTSamples transcriptions (csv, real specialties)        -> load via mtsamples=
        the ONLY source of Orthopedics; covers all 5 buckets.
  Recommended: pass BOTH -> 14k carries 4 categories, MTSamples supplies Orthopedics.

Public helpers: load_data(), expand_acronyms(), clean_text(), CATEGORIES
"""

import re, html, unicodedata, hashlib
import numpy as np
import pandas as pd

CATEGORIES = ["Cardiology", "Neurology", "Orthopedics", "Gastroenterology", "Other"]

# --- 14k medical-text corpus: numeric label 1-5 -> category ---
# cancer (neoplasms) and general both fold into "Other" per the team's decision.
MEDICAL_TEXT_MAP = {
    1: "Other",             # neoplasms / cancer  -> grouped into Other
    2: "Gastroenterology",  # digestive
    3: "Neurology",         # nervous system
    4: "Cardiology",        # cardiovascular
    5: "Other",             # general pathological
}

# --- MTSamples specialty -> category (unmapped specialties -> "Other") ---
SPECIALTY_MAP = {
    "Cardiovascular / Pulmonary": "Cardiology",   # NOTE: lumps LUNG cases into Cardiology
    "Neurology": "Neurology",
    "Neurosurgery": "Neurology",
    "Orthopedic": "Orthopedics",
    "Gastroenterology": "Gastroenterology",
}

# --------------------------------------------------------------------------
# Clinical acronym expansion (toggle; A/B test against OOF macro-F1)
# --------------------------------------------------------------------------
ACRONYMS = {
    "MI": "myocardial infarction", "EF": "ejection fraction",
    "CHF": "congestive heart failure", "CAD": "coronary artery disease",
    "CABG": "coronary artery bypass graft", "HTN": "hypertension",
    "ACS": "acute coronary syndrome", "PCI": "percutaneous coronary intervention",
    "AFib": "atrial fibrillation", "LVH": "left ventricular hypertrophy",
    "DVT": "deep vein thrombosis", "SOB": "shortness of breath",
    "COPD": "chronic obstructive pulmonary disease", "CXR": "chest x-ray",
    "CVA": "cerebrovascular accident", "TIA": "transient ischemic attack",
    "SAH": "subarachnoid hemorrhage", "ICH": "intracerebral hemorrhage",
    "LOC": "loss of consciousness", "ICP": "intracranial pressure",
    "GERD": "gastroesophageal reflux disease", "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome", "UC": "ulcerative colitis",
    "GIB": "gastrointestinal bleeding", "EGD": "esophagogastroduodenoscopy",
    "PUD": "peptic ulcer disease", "LFT": "liver function test",
    "ORIF": "open reduction internal fixation", "ROM": "range of motion",
    "DJD": "degenerative joint disease", "OA": "osteoarthritis",
    "DM": "diabetes mellitus", "UTI": "urinary tract infection",
    "CKD": "chronic kidney disease", "ESRD": "end stage renal disease",
    "Hx": "history", "Dx": "diagnosis", "Tx": "treatment", "Sx": "symptoms",
    "Fx": "fracture", "Bx": "biopsy",
}
_ACR_KEYS = sorted(ACRONYMS, key=len, reverse=True)
_ACR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in _ACR_KEYS) + r")(?![A-Za-z])")


def expand_acronyms(text):
    return _ACR_RE.sub(lambda m: ACRONYMS[m.group(1)], text)


def clean_text(t):
    t = html.unescape(str(t))
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if ch in "\n\t" or ord(ch) >= 32)
    return re.sub(r"\s+", " ", t).strip()


def _dedup_key(t):
    k = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return hashlib.md5(re.sub(r"\s+", " ", k).strip().encode()).hexdigest()


# ---- raw loaders: each returns a list of (raw_text, category) pairs ----
def _load_medical_text(path):
    pairs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "\t" not in ln:
                continue
            lab, txt = ln.rstrip("\n").split("\t", 1)
            cat = MEDICAL_TEXT_MAP.get(int(lab.strip()), "Other")
            pairs.append((txt, cat))
    return pairs


def _load_mtsamples(path, text_col="transcription"):
    df = pd.read_csv(path)
    pairs = []
    for _, row in df.iterrows():
        raw = row.get(text_col, "")
        if pd.isna(raw):
            raw = row.get("description", "")
        spec = re.sub(r"\s+", " ", str(row.get("medical_specialty", ""))).strip()
        pairs.append((raw, SPECIALTY_MAP.get(spec, "Other")))
    return pairs


def load_data(medtext=None, mtsamples=None, expand=False, dedup=True, min_chars=40):
    """
    Build the training set from one or both sources.
    Returns (texts: list[str], labels: np.ndarray[str]).
    clean -> (optional) acronym-expand -> drop too-short -> dedup ACROSS all sources.
    """
    pairs = []
    if medtext:
        pairs += _load_medical_text(medtext)
    if mtsamples:
        pairs += _load_mtsamples(mtsamples)
    if not pairs:
        raise ValueError("pass medtext=, mtsamples=, or both")

    # Label-aware dedup ordering: MTSamples lists the same transcription under
    # multiple specialties. Process named categories BEFORE "Other" (stable sort)
    # so a note duplicated across e.g. Orthopedic + Surgery is kept as Orthopedics,
    # not stolen into Other.
    pairs.sort(key=lambda p: p[1] == "Other")

    texts, labels, seen = [], [], set()
    for raw, cat in pairs:
        t = clean_text(raw)
        if expand:
            t = expand_acronyms(t)
        if len(t) < min_chars:
            continue
        if dedup:
            k = _dedup_key(t)
            if k in seen:
                continue
            seen.add(k)
        texts.append(t)
        labels.append(cat)
    return texts, np.array(labels)


if __name__ == "__main__":
    import argparse, collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--medtext")
    ap.add_argument("--mtsamples")
    ap.add_argument("--expand", action="store_true")
    a = ap.parse_args()
    if not (a.medtext or a.mtsamples):
        print("acronym demo:", expand_acronyms(clean_text("Pt with SOB, low EF, hx of MI.")))
    else:
        X, y = load_data(medtext=a.medtext, mtsamples=a.mtsamples, expand=a.expand)
        print(f"loaded {len(X)} notes after dedup (expand={a.expand})")
        print("category counts:", dict(sorted(collections.Counter(y.tolist()).items())))
