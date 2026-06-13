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
numeric_to_condition = {1: "Cardiovascular diseases", 2: "Digestive system diseases",
                        3: "Nervous system diseases", 4: "Neoplasms",
                        5: "General pathological conditions"}
label_map_1 = {
    "Cardiovascular diseases": "Cardiology",
    "Nervous system diseases": "Neurology",
    "Digestive system diseases": "Gastroenterology",
    "Neoplasms": "Other_Cancer",           # Keep separate internally
    "General pathological conditions": "Other_General",  # Keep separate internally
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

def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z0-9\s]", "", text).strip()

df["clinical_text"] = df["clinical_text"].apply(clean_text)
df = df[df["clinical_text"].str.len() > 0].reset_index(drop=True)

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
