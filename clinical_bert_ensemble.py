import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


# =============================================================================
# Phase 1 & 2: GPU-Accelerated ClinicalBERT Feature Extraction
# =============================================================================

def extract_clinical_bert_embeddings(texts, batch_size=32):
    """
    Treats Bio_ClinicalBERT strictly as a frozen feature extractor.
    Uses mean pooling across the sequence length (ignoring padding tokens).
    Returns a dense numpy matrix of shape (n_samples, 768).
    """
    # Lazy imports: keep torch/transformers out of the module-level namespace
    # so the ensemble process (XGBoost) never loads MPS/CUDA libraries.
    import torch
    from transformers import AutoTokenizer, AutoModel

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} for feature extraction")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True
    ).to(device)
    model.eval()

    all_embeddings = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, len(texts))} of {len(texts)})",
                  end="\r", flush=True)
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(device)

            outputs = model(**inputs)

            # Average last 4 hidden layers for richer representations
            last_4 = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)

            # Mean pooling (ignoring padding tokens)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            summed = (last_4 * attention_mask).sum(1)
            counts = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()

            all_embeddings.append(mean_pooled)

    print()  # newline after progress bar
    result = np.vstack(all_embeddings)

    # Free MPS/GPU memory before returning — prevents segfault when
    # a second heavy process (e.g. XGBoost) runs in the same Python session.
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =============================================================================
# Phase 2: Leakage-Free TF-IDF Extraction
# =============================================================================

def build_clinical_tfidf(train_texts, val_texts, test_texts=None):
    """
    Initializes a strict TF-IDF vectorizer to control feature space explosion
    and applies it ONLY to training data, then transforms validation/test data.
    """
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),        # Catch unigrams, bigrams, and trigrams
        sublinear_tf=True,         # Logarithmic scaling: TF -> 1 + log(TF)
        max_features=3000,         # STRICT CAP: Prevents the feature space from exploding
        min_df=3,                  # Ignore words/phrases that appear in fewer than 3 notes
        max_df=0.60,               # Strip words in >60% of notes (boilerplate like "patient", "history")
        stop_words='english'       # Strip generic English filler words
    )
    
    # Fit ONLY on the training fold to prevent data leakage
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    
    # Transform validation (and hidden test set) using the vocabulary learned from train
    X_val_tfidf = vectorizer.transform(val_texts)
    
    if test_texts is not None:
        X_test_tfidf = vectorizer.transform(test_texts)
        return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
        
    return X_train_tfidf, X_val_tfidf, vectorizer


def optimize_threshold_weights(oof_probabilities, true_labels, n_classes=5):
    """Finds per-class probability multipliers that maximize Macro F1."""
    def loss_func(weights):
        preds = np.argmax(oof_probabilities * weights, axis=1)
        return -f1_score(true_labels, preds, average='macro')

    result = minimize(loss_func, [1.0] * n_classes,
                      method='Nelder-Mead', bounds=[(0.1, 3.0)] * n_classes)
    return result.x


# =============================================================================
# Phase 3: Complete Pipeline Orchestrator
# Feature Alignment, Leakage-Free CV, and Soft-Voting Ensemble Blend
# =============================================================================

def run_pipeline(raw_texts, bert_embeddings, labels, test_texts=None, test_bert=None, n_classes=5, class_names=None):
    """
    Complete Phase 3 Orchestrator.
    
    Args:
        raw_texts:       List of raw clinical note strings (~1,000).
        bert_embeddings: Pre-computed dense array of shape (n_samples, 768).
        labels:          Numpy array of target labels (0 to n_classes-1).
        test_texts:      Optional list of hidden test set clinical strings.
        test_bert:       Optional pre-computed dense array for the test set.
        n_classes:       Number of target classes (default 5).
    
    Returns:
        oof_predictions:       Array of OOF predicted labels.
        final_test_predictions: Array of test set predicted labels (or None).
    """
    # Ensure inputs are numpy arrays for index slicing
    raw_texts = np.array(raw_texts)
    bert_embeddings = np.array(bert_embeddings)
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track out-of-fold predictions for overall evaluation
    oof_predictions = np.zeros(len(raw_texts), dtype=int)
    oof_probabilities = np.zeros((len(raw_texts), n_classes))
    
    # Accumulate test predictions across folds for averaging
    test_preds_accumulated = []
    
    fold_f1_scores = []
    
    print("Starting 5-Fold Stratified Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_texts, labels)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        
        # ----- Step A & B: Split raw text and labels -----
        train_text_fold = raw_texts[train_idx]
        val_text_fold = raw_texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # ----- Step C: Fit TF-IDF on training fold only (No Leakage) -----
        if test_texts is not None:
            X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = build_clinical_tfidf(
                train_text_fold, val_text_fold, test_texts
            )
        else:
            X_train_tfidf, X_val_tfidf, vectorizer = build_clinical_tfidf(
                train_text_fold, val_text_fold
            )
        
        # ----- Step D: Slice pre-computed BERT embeddings -----
        X_train_bert = csr_matrix(bert_embeddings[train_idx])
        X_val_bert = csr_matrix(bert_embeddings[val_idx])
        
        # ----- Step E: Feature Fusion (hstack + tocsr) -----
        X_train_combined = hstack([X_train_tfidf, X_train_bert]).tocsr()
        X_val_combined = hstack([X_val_tfidf, X_val_bert]).tocsr()
        
        # ----- Initialize Regularized Models -----
        # Build weights keyed on encoded label integers — safe regardless of LE order
        from sklearn.preprocessing import LabelEncoder as _LE
        _le = _LE(); _le.fit(np.unique(labels))
        _weight_map = {"Other": 2.0, "Neurology": 1.2, "Orthopedics": 1.2}
        _cw = {int(i): _weight_map.get(c, 1.0) for i, c in enumerate(_le.classes_)}
        model_lr = LogisticRegression(
            C=0.05, max_iter=1000, class_weight=_cw, random_state=42
        )
        model_xgb = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.03, n_estimators=400,
            subsample=0.75, colsample_bytree=0.7, min_child_weight=3,
            random_state=42, early_stopping_rounds=15
        )

        # ----- Train Meta-Classifiers -----
        model_lr.fit(X_train_combined, y_train)
        model_xgb.fit(X_train_combined, y_train,
                      eval_set=[(X_val_combined, y_val)], verbose=False)
        
        # ----- Soft-Voting Probability Blending -----
        prob_lr = model_lr.predict_proba(X_val_combined)
        prob_xgb = model_xgb.predict_proba(X_val_combined)
        
        # Dynamic blend: find optimal LR weight on this validation fold
        best_f1, best_weight = 0, 0.5
        for w in np.linspace(0.1, 0.9, 9):
            preds = np.argmax((w * prob_lr) + ((1 - w) * prob_xgb), axis=1)
            score = f1_score(y_val, preds, average='macro')
            if score > best_f1:
                best_f1, best_weight = score, w
        print(f"  Best blend weight (LR): {best_weight:.2f}")

        blended_prob = (best_weight * prob_lr) + ((1 - best_weight) * prob_xgb)
        oof_probabilities[val_idx] = blended_prob
        oof_predictions[val_idx] = np.argmax(blended_prob, axis=1)
        
        fold_f1 = f1_score(y_val, oof_predictions[val_idx], average='macro')
        fold_f1_scores.append(fold_f1)
        print(f"Fold {fold + 1} Macro F1: {fold_f1:.4f}")
        
        # ----- Hidden Test Set Inference (if provided) -----
        if test_texts is not None and test_bert is not None:
            X_test_bert = csr_matrix(test_bert)
            X_test_combined = hstack([X_test_tfidf, X_test_bert]).tocsr()
            
            prob_test_lr = model_lr.predict_proba(X_test_combined)
            prob_test_xgb = model_xgb.predict_proba(X_test_combined)
            test_blended_prob = (best_weight * prob_test_lr) + ((1 - best_weight) * prob_test_xgb)
            test_preds_accumulated.append(test_blended_prob)

    # =================================================================
    # Threshold Calibration: find per-class weights on OOF probabilities
    # =================================================================
    print("Optimizing threshold weights on OOF probabilities...")
    threshold_weights = optimize_threshold_weights(oof_probabilities, labels, n_classes)
    print(f"Calibrated weights: {np.round(threshold_weights, 3)}")
    oof_predictions = np.argmax(oof_probabilities * threshold_weights, axis=1)

    # =================================================================
    # Overall Out-Of-Fold Evaluation
    # =================================================================
    overall_f1 = f1_score(labels, oof_predictions, average='macro')
    print("\n" + "="*50)
    print(f"Average Fold Macro F1: {np.mean(fold_f1_scores):.4f} (± {np.std(fold_f1_scores):.4f})")
    print(f"OVERALL OOF MACRO F1:  {overall_f1:.4f}")
    print("="*50 + "\n")
    print(classification_report(labels, oof_predictions, target_names=class_names, digits=2))

    if class_names is not None:
        import pandas as pd
        out = pd.DataFrame({"id": np.arange(len(labels)), "true": labels, "pred": oof_predictions})
        for j, c in enumerate(class_names):
            out[f"prob_{c}"] = oof_probabilities[:, j]
        out.to_csv("oof_predictions.csv", index=False)
        print("saved OOF table -> oof_predictions.csv")
    
    # Average test probabilities across all 5 folds for final submission
    if test_preds_accumulated:
        final_test_probabilities = np.mean(test_preds_accumulated, axis=0)
        return oof_predictions, np.argmax(final_test_probabilities * threshold_weights, axis=1)
        
    return oof_predictions, None


# =============================================================================
# Mock Test
# =============================================================================

if __name__ == "__main__":
    print("Running example test to verify the full pipeline...\n")
    
    # Generate mock texts and multi-class target labels (5 classes: 0-4)
    sample_texts = [
        "Patient has a history of hypertension and diabetes.",
        "The patient is presenting with acute abdominal pain.",
        "No known allergies. Blood pressure is normal.",
        "Prescribed lisinopril 10mg daily for heart failure.",
        "Patient reported feeling dizzy and short of breath.",
        "Diagnosed with stage 2 chronic kidney disease.",
        "Post-operative recovery from knee replacement surgery.",
        "Patient complains of persistent lower back pain.",
        "Lab results show elevated white blood cell count.",
        "Follow-up visit for management of Type 2 diabetes.",
    ] * 5  # 50 samples so min_df=3 can find overlapping terms
    
    y_mock = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 5)
    
    # Phase 1: Extract BERT embeddings (run once, reuse forever)
    print("Extracting ClinicalBERT embeddings for mock data...")
    X_bert_mock = extract_clinical_bert_embeddings(sample_texts, batch_size=8)
    print(f"BERT embeddings shape: {X_bert_mock.shape}\n")
    
    # Phase 3: Run the full pipeline
    oof_preds, test_preds = run_pipeline(sample_texts, X_bert_mock, y_mock)
    print("Pipeline complete!")
