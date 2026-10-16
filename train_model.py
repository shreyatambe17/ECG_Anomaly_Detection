
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
DATA_DIR = Path("data") / "mitdb"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRE_SAMPLES = 90
POST_SAMPLES = 126
SEGMENT_LENGTH = PRE_SAMPLES + POST_SAMPLES

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

ABNORMAL_SYMBOLS = {
    "A", "a", "J", "S", "V", "E",
    "F", "/", "f", "Q"
}

# Keep training, validation, and testing recordings separate.
TRAIN_RECORDS = [
    "102", "103", "104", "105", "107",
    "109", "111", "112", "113", "114", "115"
]

VALIDATION_RECORDS = ["106", "108", "117"]
TEST_RECORDS = ["100", "101"]

# Thresholds are compared using validation data only.
THRESHOLDS = np.arange(0.05, 0.51, 0.05)


# ==========================================
# 2. LOAD AND PREPARE ECG HEARTBEATS
# ==========================================

def load_segments(record_names):
    X = []
    y = []

    for record_name in record_names:
        print(f"\nPreparing ECG record {record_name}...")

        record_path = DATA_DIR / record_name
        signal = None
        annotation = None

        # Retry interrupted downloads up to three times.
        for attempt in range(1, 4):
            try:
                header_file = Path(str(record_path) + ".hea")

                signal_files = [
                    p for p in DATA_DIR.glob(record_name + ".*")
                    if p.suffix in {".dat", ".mat"}
                ]

                if not header_file.exists() or not signal_files:
                    print(
                        f"Downloading {record_name} "
                        f"(attempt {attempt}/3)..."
                    )

                    wfdb.dl_database(
                        "mitdb",
                        dl_dir=str(DATA_DIR),
                        records=[record_name]
                    )

                signal, _fields = wfdb.rdsamp(str(record_path))
                annotation = wfdb.rdann(str(record_path), "atr")

                print(f"Record {record_name} ready.")
                break

            except Exception as error:
                print(
                    f"Attempt {attempt} failed for "
                    f"{record_name}: {error}"
                )

                signal = None
                annotation = None

                # Remove incomplete files before retrying.
                for file_path in DATA_DIR.glob(record_name + ".*"):
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

                if attempt < 3:
                    wait_seconds = attempt * 3
                    print(f"Retrying in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                else:
                    raise RuntimeError(
                        f"Could not download/read record "
                        f"{record_name} after 3 attempts. "
                        "Check your internet connection and retry."
                    ) from error

        # Use the first ECG channel.
        ecg = signal[:, 0]

        # Extract a fixed-length segment around each annotation.
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

            # Normalize each heartbeat independently.
            segment_std = np.std(segment)

            if segment_std <= 1e-8:
                continue

            segment = (
                segment - np.mean(segment)
            ) / segment_std

            X.append(segment)
            y.append(label)

        print(f"Record {record_name} processed successfully.")

    if not X:
        raise ValueError("No ECG segments were loaded.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y)
    )


# ==========================================
# 3. EXTRACT ADDITIONAL ECG FEATURES
# ==========================================

def extract_features(X):
    """
    Append 18 statistical and signal-shape features
    to each 216-sample normalized ECG segment.
    """
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

    # Retain waveform samples and append engineered features.
    return np.hstack([X, engineered]).astype(np.float32)


# ==========================================
# 4. LOAD DATA AND APPLY FEATURES
# ==========================================

print("\nLoading training data...")
X_train, y_train = load_segments(TRAIN_RECORDS)

print("\nLoading validation data...")
X_val, y_val = load_segments(VALIDATION_RECORDS)

print("\nLoading test data...")
X_test, y_test = load_segments(TEST_RECORDS)

print("\nRaw segment shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining class counts:")
print(pd.Series(y_train).value_counts())

print("\nValidation class counts:")
print(pd.Series(y_val).value_counts())

print("\nTesting class counts:")
print(pd.Series(y_test).value_counts())

print("\nExtracting ECG features...")

X_train = extract_features(X_train)
X_val = extract_features(X_val)
X_test = extract_features(X_test)

print("Feature extraction completed.")
print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)
print("Testing feature shape:", X_test.shape)


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
print("Training completed!")


# ==========================================
# 6. SELECT THRESHOLD USING VALIDATION DATA
# ==========================================

class_names = list(model.classes_)
abnormal_index = class_names.index("Abnormal")

val_prob = model.predict_proba(X_val)[:, abnormal_index]

best_threshold = 0.50
best_f1 = -1.0

print("\nValidation threshold comparison:")

for threshold in THRESHOLDS:
    val_pred = np.where(
        val_prob >= threshold,
        "Abnormal",
        "Normal"
    )

    abnormal_f1 = f1_score(
        y_val,
        val_pred,
        pos_label="Abnormal",
        zero_division=0
    )

    abnormal_recall = recall_score(
        y_val,
        val_pred,
        pos_label="Abnormal",
        zero_division=0
    )

    val_balanced_accuracy = balanced_accuracy_score(
        y_val,
        val_pred
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Abnormal F1={abnormal_f1:.4f} | "
        f"Abnormal recall={abnormal_recall:.4f} | "
        f"Balanced accuracy={val_balanced_accuracy:.4f}"
    )

    # Choose the threshold with the highest validation F1.
    if abnormal_f1 > best_f1:
        best_f1 = abnormal_f1
        best_threshold = float(threshold)

print(f"\nSelected validation threshold: {best_threshold:.2f}")


# ==========================================
# 7. FINAL EVALUATION ON TEST DATA
# ==========================================

# The test set is not used to choose the threshold.
test_prob = model.predict_proba(X_test)[:, abnormal_index]

y_pred = np.where(
    test_prob >= best_threshold,
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

print("\nFinal test results:")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Abnormal recall: {abnormal_recall:.4f}")
print(f"Abnormal F1-score: {abnormal_f1:.4f}")

print("\nClassification report:")
print(report)

print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)


# ==========================================
# 8. SAVE MODEL AND RESULTS
# ==========================================

model_bundle = {
    "model": model,
    "pre_samples": PRE_SAMPLES,
    "post_samples": POST_SAMPLES,
    "segment_length": SEGMENT_LENGTH,
    "class_names": ["Normal", "Abnormal"],
    "input_channel": 0,
    "threshold": best_threshold,
    "normalization": "per_heartbeat_z_score",
    "feature_engineering": "raw_waveform_plus_18_features",
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "ecg_random_forest.joblib"
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

with open(
    OUTPUT_DIR / "metrics.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
    f.write(f"Abnormal recall: {abnormal_recall:.4f}\n")
    f.write(f"Abnormal F1-score: {abnormal_f1:.4f}\n")
    f.write(f"Selected threshold: {best_threshold:.2f}\n")
    f.write(f"Confusion matrix:\n{cm}\n")

# Plot the final test confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Abnormal"])
ax.set_yticklabels(["Normal", "Abnormal"])
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_title("ECG Classification Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=150
)

plt.close(fig)

print("\nModel saved successfully!")
print("Model: models/ecg_random_forest.joblib")
print("Results: outputs/")
