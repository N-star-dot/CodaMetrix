"""
Unleash System Diagnostic
Runs BERT extraction in a subprocess to avoid MPS/XGBoost thread conflict on Apple Silicon.
"""
import subprocess
import sys
import numpy as np
import pandas as pd


def generate_dummy_clinical_data(num_samples=50):
    np.random.seed(42)
    medical_phrases = [
        "Patient presents with severe shortness of breath and chest pain. EKG shows elevated ST.",
        "Frequent headaches, dizziness, and mild ataxia noted over the last 3 weeks.",
        "Compound fracture of the left femur, requires immediate surgical fixation.",
        "Reports severe abdominal pain, nausea, and history of severe acid reflux.",
        "Routine checkup, no acute distress, all vaccinations updated today."
    ]
    raw_texts = np.random.choice(medical_phrases, size=num_samples)
    labels = np.random.randint(0, 5, size=num_samples)
    test_texts = np.random.choice(medical_phrases, size=10)
    print(f"Generated {num_samples} training samples and 10 test samples.")
    return list(raw_texts), labels, list(test_texts)


# Subprocess script: extracts embeddings and saves to .npy files
_EXTRACT_SCRIPT = """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[4])
from clinical_bert_ensemble import extract_clinical_bert_embeddings

data = json.load(open(sys.argv[1]))
train_emb = extract_clinical_bert_embeddings(data['train_texts'], batch_size=8)
test_emb  = extract_clinical_bert_embeddings(data['test_texts'],  batch_size=8)
np.save(sys.argv[2], train_emb)
np.save(sys.argv[3], test_emb)
print(f"Saved train {train_emb.shape} and test {test_emb.shape}")
"""


if __name__ == "__main__":
    import json, os

    print("Initializing Unleash System Diagnostic...")

    texts, labels, test_texts = generate_dummy_clinical_data(num_samples=50)

    # --- [1/3] BERT Extraction in isolated subprocess (avoids MPS/XGBoost conflict) ---
    print("\n[1/3] Testing ClinicalBERT Extraction (MPS/GPU) via subprocess...")

    payload = {"train_texts": texts, "test_texts": test_texts}
    with open("/tmp/unleash_input.json", "w") as f:
        json.dump(payload, f)
    with open("/tmp/_extract_runner.py", "w") as f:
        f.write(_EXTRACT_SCRIPT)

    result = subprocess.run(
        [sys.executable, "/tmp/_extract_runner.py",
         "/tmp/unleash_input.json",
         "/tmp/train_emb.npy",
         "/tmp/test_emb.npy",
         os.path.dirname(os.path.abspath(__file__))],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("EXTRACTION FAILED:\n", result.stderr)
        sys.exit(1)
    print(result.stdout.strip())

    train_embeddings = np.load("/tmp/train_emb.npy")
    test_embeddings  = np.load("/tmp/test_emb.npy")
    print(f"SUCCESS: Train Matrix Shape: {train_embeddings.shape}")  # (50, 768)
    print(f"SUCCESS: Test Matrix Shape:  {test_embeddings.shape}")   # (10, 768)

    # --- [2/3] Ensemble pipeline (XGBoost runs cleanly, no MPS in this process) ---
    print("\n[2/3] Testing Cross-Validation and Dynamic Blended Ensemble...")
    from clinical_bert_ensemble import run_pipeline
    oof_preds, test_preds = run_pipeline(
        raw_texts=texts,
        bert_embeddings=train_embeddings,
        labels=labels,
        test_texts=test_texts,
        test_bert=test_embeddings
    )

    # --- [3/3] Submission export ---
    print("\n[3/3] Simulating Final CSV Export...")
    submission = pd.DataFrame({
        'note_id': [f"TEST_{i}" for i in range(len(test_preds))],
        'predicted_specialty': test_preds
    })
    submission.to_csv("unleash_diagnostic_submission.csv", index=False)
    print("\nDiagnostic Complete. Pipeline is stable and ready for the real dataset.")

    # Cleanup temp files
    for f in ["/tmp/unleash_input.json", "/tmp/_extract_runner.py",
              "/tmp/train_emb.npy", "/tmp/test_emb.npy"]:
        os.remove(f)
