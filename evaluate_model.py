
from pathlib import Path

import joblib
import numpy as np
import wfdb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/ecg_random_forest.joblib")
DATA_DIR = Path("data") / "mitdb"

TEST_RECORDS = ["100", "101"]

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

ABNORMAL_SYMBOLS = {
    "A", "a", "J", "S", "V", "E",
    "F", "/", "f", "Q"
}


# ==========================================
# 2. LOAD THE SAVED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. "
        "Run python train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
PRE_SAMPLES = bundle["pre_samples"]
POST_SAMPLES = bundle["post_samples"]
SEGMENT_LENGTH = bundle["segment_length"]
THRESHOLD = bundle["threshold"]

print("Saved ECG model loaded successfully.")
print(f"Decision threshold: {THRESHOLD:.2f}")


# ==========================================
# 3. LOAD TEST ECG SEGMENTS
# ==========================================

def load_test_segments(record_names):
    X = []
    y = []

    for record_name in record_names:
        record_path = DATA_DIR / record_name

        if not Path(str(record_path) + ".hea").exists():
            raise FileNotFoundError(
                f"ECG record {record_name} is missing. "
                "Run python train_model.py to download it."
            )

        print(f"\nEvaluating ECG record {record_name}...")

        signal, _fields = wfdb.rdsamp(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")

        ecg = signal[:, 0]

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):
            if symbol in NORMAL_SYMBOLS:
                label = "Normal"
            elif symbol in ABNORMAL_SYMBOLS:
                label = "Abnormal"
            else:
                continue

            start = sample - PRE_SAMPLES
            end = sample + POST_SAMPLES

            if start < 0 or end > len(ecg):
                continue

            segment = ecg[start:end].astype(np.float32)

            if len(segment) != SEGMENT_LENGTH:
                continue

            # Apply the same normalization used during training.
            if bundle.get("normalization") == "per_heartbeat_z_score":
                segment_std = np.std(segment)

                if segment_std <= 1e-8:
                    continue

                segment = (
                    segment - np.mean(segment)
                ) / segment_std

            X.append(segment)
            y.append(label)

    if not X:
        raise ValueError("No test ECG segments were loaded.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y)
    )


# ==========================================
# 4. APPLY THE SAME FEATURE ENGINEERING
# ==========================================

def extract_features(X):
    """Generate the same 18 features used during training."""
    engineered = []

    for segment in X:
        segment = np.asarray(segment, dtype=np.float32)
        differences = np.diff(segment)

        mean_value = np.mean(segment)
        std_value = np.std(segment) + 1e-8
        centered = segment - mean_value

        skewness = np.mean((centered / std_value) ** 3)
        kurtosis = np.mean((centered / std_value) ** 4) - 3

        features = [
            np.mean(segment),
            np.std(segment),
            np.min(segment),
            np.max(segment),
            np.ptp(segment),
            np.median(segment),
            np.percentile(segment, 10),
            np.percentile(segment, 25),
            np.percentile(segment, 75),
            np.percentile(segment, 90),
            np.sqrt(np.mean(segment ** 2)),
            np.mean(np.abs(differences)),
            np.std(differences),
            np.max(np.abs(differences)),
            np.sum(segment ** 2),
            np.mean(np.diff(np.signbit(segment))),
            skewness,
            kurtosis,
        ]

        engineered.append(features)

    engineered = np.asarray(engineered, dtype=np.float32)

    return np.hstack([X, engineered]).astype(np.float32)


# ==========================================
# 5. PREDICT AND EVALUATE
# ==========================================

X_test, y_test = load_test_segments(TEST_RECORDS)

# Apply features only when required by the saved model.
if bundle.get("feature_engineering") == "raw_waveform_plus_18_features":
    X_test = extract_features(X_test)

print("\nTest feature shape:", X_test.shape)

class_names = list(model.classes_)
abnormal_index = class_names.index("Abnormal")

probabilities = model.predict_proba(X_test)[:, abnormal_index]

y_pred = np.where(
    probabilities >= THRESHOLD,
    "Abnormal",
    "Normal"
)

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

abnormal_recall = recall_score(
    y_test,
    y_pred,
    pos_label="Abnormal",
    zero_division=0
)

abnormal_f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Abnormal",
    zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=["Normal", "Abnormal"],
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Normal", "Abnormal"]
)

print("\n========== ECG MODEL EVALUATION ==========")
print("Test records:", ", ".join(TEST_RECORDS))
print("Number of test segments:", len(y_test))

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Abnormal recall: {abnormal_recall:.4f}")
print(f"Abnormal F1-score: {abnormal_f1:.4f}")

print("\nClassification report:")
print(report)

print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)

print("\nEvaluation completed.")
