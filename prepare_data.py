import re
import os
import pandas as pd
import kagglehub

# =============================================================================
# Step 0: Download datasets
# =============================================================================
print("Downloading datasets...")
path_1 = kagglehub.dataset_download("chaitanyakck/medical-text")
path_2 = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
print(f"Source 1: {path_1}")
print(f"Source 2: {path_2}")

# =============================================================================
# Step 1: Load & standardize Source 1 (medical-text, tab-separated .dat)
# =============================================================================
# Label mapping: numeric code -> condition name -> target class
# NOTE: codes verified against the note text — 1 is cancer, 4 is cardiac
# (label-1 note = "cancer patients..."; label-4 note = "myocardial infarction...").
numeric_to_condition = {1: "Neoplasms", 2: "Digestive system diseases",
                        3: "Nervous system diseases", 4: "Cardiovascular diseases",
                        5: "General pathological conditions"}
label_map_1 = {
    "Cardiovascular diseases": "Cardiology",
    "Nervous system diseases": "Neurology",
    "Digestive system diseases": "Gastroenterology",
    "Neoplasms": "Other",
    "General pathological conditions": "Other",
}

df1 = pd.read_csv(os.path.join(path_1, "train.dat"), sep="\t", header=None,
                  names=["condition_label", "clinical_text"])
df1["condition_label"] = df1["condition_label"].map(numeric_to_condition)
df1["target_label"] = df1["condition_label"].map(label_map_1)
df1 = df1[["clinical_text", "target_label"]].dropna()
print(f"Source 1 shape: {df1.shape}")
print(df1["target_label"].value_counts())

# =============================================================================
# Step 1: Load & standardize Source 2 (medicaltranscriptions)
# =============================================================================
df2 = pd.read_csv(os.path.join(path_2, "mtsamples.csv"))
print(f"\nSource 2 columns: {df2.columns.tolist()}")

df2 = df2.rename(columns={"transcription": "clinical_text"}).dropna(subset=["clinical_text"])
df2 = df2[df2["medical_specialty"].str.contains("orthopedic", case=False, na=False)].copy()
df2["target_label"] = "Orthopedics"
df2 = df2[["clinical_text", "target_label"]]

# =============================================================================
# Step 2: Concatenate & sanitize
# =============================================================================
df = pd.concat([df1, df2], ignore_index=True)

# Same cleaning as data.py / parse_cases.py: keeps case + punctuation (no lowercasing,
# no punctuation stripping) so clinical acronyms survive — then expand them. Baking the
# expansion into the CSV means his training path always gets it, and it matches the
# test-file preprocessing (predict.py expands the cases too).
from data import clean_text, expand_acronyms

df["clinical_text"] = df["clinical_text"].apply(lambda t: expand_acronyms(clean_text(t)))
df = df.drop_duplicates(subset="clinical_text")                      # no dup -> no CV leakage
df = df[df["clinical_text"].str.len() >= 40].reset_index(drop=True)  # match data.py min_chars

print(f"\nCombined dataset shape: {df.shape}")
print(df["target_label"].value_counts())

# =============================================================================
# Step 3: Stratified downsample to 1,000 rows (200 per class)
# =============================================================================
classes = ["Cardiology", "Neurology", "Gastroenterology", "Orthopedics", "Other"]
subsets = []
for cls in classes:
    pool = df[df["target_label"] == cls]
    if len(pool) < 200:
        raise ValueError(f"Not enough rows for class '{cls}': only {len(pool)} available.")
    subsets.append(pool.sample(200, random_state=42))

final = pd.concat(subsets).sample(frac=1, random_state=42).reset_index(drop=True)
final.to_csv("unleash_train_1k.csv", index=False)

print(f"\nFinal shape: {final.shape}")
print(final["target_label"].value_counts())
print("\nunleash_train_1k.csv saved successfully.")
