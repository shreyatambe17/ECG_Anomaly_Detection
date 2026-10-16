from pathlib import Path
from io import BytesIO, StringIO
import base64

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, render_template_string


# ==================================================
# 1. APP CONFIGURATION
# ==================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ecg_random_forest.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
THRESHOLD = float(bundle["threshold"])
EXPECTED_SAMPLES = int(bundle.get("segment_length", 216))
FEATURE_ENGINEERING = bundle.get("feature_engineering", "")

MAX_SEGMENTS = 500


# ==================================================
# 2. FEATURE EXTRACTION
# Must match train_model.py
# ==================================================

def extract_features(segment):
    segment = np.asarray(segment, dtype=np.float32)
    differences = np.diff(segment)

    mean = np.mean(segment)
    std = np.std(segment)

    if std > 1e-8:
        z = (segment - mean) / std
        skewness = np.mean(z ** 3)
        kurtosis = np.mean(z ** 4) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    engineered = np.array([
        mean,
        std,
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
    ], dtype=np.float32)

    return np.concatenate([segment, engineered]).astype(np.float32)


# ==================================================
# 3. ECG WAVEFORM
# ==================================================

def create_waveform(signal):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(np.arange(len(signal)), signal, linewidth=1.5)
    ax.set_title("ECG Segment 1 (Normalized)")
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    image_buffer = BytesIO()
    fig.savefig(image_buffer, format="png", dpi=120)
    plt.close(fig)
    image_buffer.seek(0)

    return base64.b64encode(
        image_buffer.read()
    ).decode("utf-8")

# ==================================================
# 4. DASHBOARD UI
# ==================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECG Insight | ECG Analysis Dashboard</title>

<style>
:root {
    --navy: #082448;
    --navy2: #123e66;
    --teal: #08aaa5;
    --teal2: #087f87;
    --blue: #188bdb;
    --ink: #10244b;
    --muted: #61769d;
    --line: #dce8f5;
    --pale: #edf6fc;
    --white: #fff;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    color: var(--ink);
    background: linear-gradient(
        120deg,
        #f3faff 0%,
        #eaf4fb 55%,
        #f7fbff 100%
    );
    font-size: 14px;
}

.topbar {
    min-height: 74px;
    background: linear-gradient(
        100deg,
        #071c38,
        #123a60 65%,
        #0a2547
    );
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 3.6%;
    gap: 20px;
    position: relative;
    overflow: hidden;
}

.topbar:after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(
            ellipse at 42% 20%,
            rgba(43,193,220,.18),
            transparent 28%
        ),
        radial-gradient(
            ellipse at 78% 70%,
            rgba(20,185,207,.14),
            transparent 25%
        );
    pointer-events: none;
}

.brand,
.nav,
.brandtext {
    display: flex;
    align-items: center;
    position: relative;
    z-index: 1;
}

.brand {
    gap: 13px;
    min-width: 290px;
}

.brandmark {
    width: 52px;
    height: 52px;
    display: grid;
    place-items: center;
    color: white;
    font-size: 33px;
    border-radius: 50%;
    border: 2px solid #76f5f2;
    background: linear-gradient(145deg, #19d1c8, #086da0);
    box-shadow: 0 0 22px #16c8d766;
}

.brandtext {
    display: block;
}

.brand h1 {
    font-size: 23px;
    margin: 0;
    line-height: 1.2;
}

.brand p {
    font-size: 14px;
    margin: 4px 0 0;
    color: #d5e6f7;
}

.nav {
    gap: 8px;
    flex: 1;
    justify-content: center;
}

.nav a {
    color: white;
    text-decoration: none;
    padding: 12px 14px;
    border-radius: 11px;
    font-size: 13px;
    white-space: nowrap;
}

.nav a.active {
    background: linear-gradient(120deg, #087e9d, #087e91);
    box-shadow: 0 7px 18px #021b3266;
}

.projecttag {
    border: 1px solid #d1e9ff;
    border-radius: 30px;
    padding: 11px 17px;
    font-size: 12px;
    white-space: nowrap;
    position: relative;
    z-index: 1;
}

main {
    max-width: 1500px;
    margin: 0 auto;
    padding: 24px 3.6% 22px;
}

.hero {
    position: relative;
    min-height: 108px;
    margin-bottom: 17px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
    overflow: hidden;
}

.hero:before {
    content: "";
    position: absolute;
    right: 0;
    top: -55px;
    width: 50%;
    height: 210px;
    background:
        radial-gradient(
            ellipse at 70% 50%,
            rgba(17,171,208,.16),
            transparent 60%
        ),
        linear-gradient(
            90deg,
            transparent,
            rgba(109,208,227,.08)
        );
    pointer-events: none;
}

.hero h2 {
    font-size: clamp(27px, 3vw, 40px);
    line-height: 1.15;
    margin: 0 0 7px;
    font-weight: 800;
    letter-spacing: -.8px;
}

.hero p {
    font-size: 16px;
    line-height: 1.45;
    color: #405c88;
    margin: 0;
    max-width: 750px;
}

.hero-badge {
    position: relative;
    z-index: 1;
    background: linear-gradient(130deg, #4e83b8, #17638d);
    color: white;
    border-radius: 16px;
    padding: 15px 19px;
    line-height: 1.5;
    font-weight: 700;
    min-width: 150px;
    box-shadow: 0 8px 20px #124d7b24;
}

.stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 13px;
    margin-bottom: 16px;
}

.stat {
    background: white;
    border: 1px solid var(--line);
    border-radius: 15px;
    min-height: 92px;
    padding: 13px 17px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 6px 20px #143d6410;
}

.stat-icon {
    width: 61px;
    height: 61px;
    flex: 0 0 61px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, #10bcb6, #07838b);
    color: white;
    font-size: 26px;
    box-shadow: 0 6px 14px #079b9b28;
}

.stat-label {
    font-size: 11px;
    color: #486493;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.stat-value {
    font-size: 19px;
    font-weight: 800;
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.stat-note {
    font-size: 12px;
    color: #6d81a4;
    margin-top: 4px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(340px, .88fr) minmax(0, 1.62fr);
    gap: 14px;
    align-items: start;
}

.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 7px 22px #143d6410;
    margin-bottom: 14px;
    min-width: 0;
}

.cardhead {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 13px 18px 8px;
}

.cardhead h3 {
    font-size: 16px;
    margin: 0;
    font-weight: 800;
}

.cardhead p {
    font-size: 12px;
    color: #6b80a4;
    margin: 5px 0 0;
}

.cardicon {
    width: 43px;
    height: 43px;
    flex: 0 0 43px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: linear-gradient(140deg, #dff7ff, #bdeeff);
    color: #087fa5;
    font-size: 23px;
}

.uploadhead {
    background: linear-gradient(105deg, #174e7b, #5db4c9);
    color: white;
    padding: 12px 19px;
}

.uploadhead .cardicon {
    background: linear-gradient(140deg, #1be0d1, #079eab);
    color: white;
}

.uploadhead p {
    color: #f0fbff;
}

.uploadbody {
    padding: 12px 17px 15px;
}

.tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    background: #f0f6fb;
    border-radius: 9px;
    padding: 4px;
    margin-bottom: 13px;
}

.tab {
    border-radius: 7px;
    padding: 9px 6px;
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    color: #193458;
}

.tab.active {
    background: linear-gradient(110deg, #0aa9a5, #078b8d);
    color: white;
}

.filebox {
    border: 1px dashed #9bc9ff;
    border-radius: 11px;
    padding: 15px;
    text-align: center;
    background: linear-gradient(180deg, #fbfdff, #f5faff);
    margin-bottom: 8px;
}

.cloud {
    font-size: 27px;
    color: #2789e5;
    margin-bottom: 5px;
}

.filebox strong {
    display: block;
    font-size: 13px;
    margin-bottom: 5px;
}

.filebox p {
    font-size: 12px;
    color: #62789e;
    line-height: 1.5;
    margin: 0 0 10px;
}

input[type=file] {
    display: block;
    width: 100%;
    font-size: 12px;
    color: #294365;
}

label {
    font-size: 12px;
    font-weight: 700;
    display: block;
    margin: 10px 0 7px;
}

textarea {
    width: 100%;
    min-height: 100px;
    resize: vertical;
    border: 1px solid #d8e6f4;
    border-radius: 9px;
    padding: 10px;
    font-family: Consolas, monospace;
    font-size: 12px;
    background: #fbfdff;
}

input:focus,
textarea:focus {
    outline: 2px solid #b5e7fa;
}

.info {
    background: #edf6ff;
    border-radius: 10px;
    padding: 11px 13px;
    color: #365783;
    font-size: 12px;
    line-height: 1.5;
    margin: 8px 0;
}

.info strong {
    color: #17365f;
}

.info ul {
    padding-left: 19px;
    margin: 4px 0 0;
}

.info li {
    margin: 2px 0;
}

button {
    width: 100%;
    border: 0;
    border-radius: 9px;
    padding: 13px 16px;
    margin-top: 8px;
    background: linear-gradient(100deg, #0caaa5, #078b86);
    color: white;
    font-size: 14px;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 6px 14px #079e9a25;
}

button:hover {
    filter: brightness(1.04);
}

.aboutbody {
    padding: 0 17px 15px;
}

.aboutbody p {
    font-size: 12px;
    line-height: 1.55;
    color: #61769b;
    margin: 0 0 10px;
}

.featuregrid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.feature {
    background: #f1f7fb;
    border-radius: 9px;
    padding: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #23436d;
}

.feature span {
    font-size: 19px;
    color: #079eb8;
}

.warning {
    background: #fff7e8;
    border: 1px solid #ffd77c;
    color: #75430b;
    border-radius: 10px;
    padding: 10px 11px;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 12px;
}

.rightcol {
    min-width: 0;
}

.wavebody {
    padding: 0 16px 13px;
}

.wavebody img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
}

.emptychart {
    height: 180px;
    border: 1px dashed #c8dced;
    border-radius: 10px;
    background: #f9fcff;
    display: grid;
    place-items: center;
    color: #7185a6;
    text-align: center;
    padding: 20px;
}

.summarygrid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    padding: 4px 16px 16px;
}

.mini {
    border-radius: 11px;
    padding: 11px 12px;
    border: 1px solid #d7e9ff;
    background: #f4f9ff;
    min-width: 0;
}

.mini.green {
    background: #effdf5;
    border-color: #b9f0ce;
}

.mini.red {
    background: #fff2f1;
    border-color: #ffc7c4;
}

.mini.pink {
    background: #fff1fa;
    border-color: #ffc4e9;
}

.mini-title {
    font-size: 11px;
    line-height: 1.35;
    color: #31517c;
}

.mini-number {
    font-size: 23px;
    font-weight: 850;
    margin-top: 7px;
    line-height: 1.15;
}

.mini-caption {
    font-size: 11px;
    color: #5d7398;
    margin-top: 5px;
    line-height: 1.35;
}

.bottomgrid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 12px;
}

.resultbody {
    padding: 0 15px 15px;
}

.predictionbox {
    border: 1px solid #b5f0cf;
    border-left: 4px solid #05c886;
    border-radius: 10px;
    background: linear-gradient(120deg, #f0fff6, #f7fff9);
    padding: 12px;
}

.predictionbox.abnormal {
    border-color: #ffc8c4;
    border-left-color: #ee574d;
    background: #fff7f6;
}

.predictiontitle {
    font-size: 14px;
    font-weight: 800;
    color: #075b43;
}

.predictionbox.abnormal .predictiontitle {
    color: #9a2424;
}

.score {
    font-size: 25px;
    font-weight: 850;
    margin: 6px 0 7px;
    color: #075b43;
}

.predictionbox.abnormal .score {
    color: #9a2424;
}

.progress {
    height: 9px;
    border-radius: 30px;
    background: #dce6ed;
    overflow: hidden;
}

.progressinner {
    height: 100%;
    border-radius: 30px;
    background: linear-gradient(90deg, #00b88d, #15a8dc);
}

.predictionnote {
    font-size: 11px;
    color: #60769a;
    line-height: 1.5;
    margin-top: 8px;
}

.tablewrap {
    padding: 0 14px 15px;
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 11px;
    min-width: 310px;
}

th, td {
    padding: 10px 8px;
    text-align: center;
    border-bottom: 1px solid #e0eaf4;
    border-right: 1px solid #e0eaf4;
}

th {
    background: #f1f6fb;
    font-size: 11px;
    color: #1b355c;
}

th:first-child,
td:first-child {
    border-left: 1px solid #e0eaf4;
}

thead th:first-child {
    border-top-left-radius: 8px;
}

thead th:last-child {
    border-top-right-radius: 8px;
}

tbody tr:last-child td:first-child {
    border-bottom-left-radius: 8px;
}

tbody tr:last-child td:last-child {
    border-bottom-right-radius: 8px;
}

.badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    background: #d4f8e2;
    color: #087440;
    font-weight: 800;
}

.badge.bad {
    background: #ffe1df;
    color: #a52b2b;
}

.error {
    margin: 0 0 14px;
    padding: 12px 14px;
    background: #fff0f0;
    border: 1px solid #ffcaca;
    border-radius: 10px;
    color: #9c2525;
    font-size: 13px;
}

footer {
    text-align: center;
    color: #7486a2;
    font-size: 11px;
    line-height: 1.6;
    padding: 5px 0 18px;
}

@media(max-width: 1100px) {
    .nav {
        display: none;
    }

    .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .layout {
        grid-template-columns: minmax(300px, .9fr) minmax(0, 1.1fr);
    }

    .summarygrid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media(max-width: 780px) {
    .topbar {
        padding: 12px 4%;
        height: auto;
        flex-wrap: wrap;
    }

    .brand {
        min-width: 0;
    }

    .projecttag {
        display: none;
    }

    main {
        padding: 18px 4%;
    }

    .layout,
    .bottomgrid {
        grid-template-columns: 1fr;
    }

    .hero-badge {
        display: none;
    }

    .hero p {
        font-size: 14px;
    }

    .stats {
        grid-template-columns: 1fr 1fr;
    }

    .stat {
        padding: 11px;
        gap: 9px;
    }

    .stat-icon {
        width: 43px;
        height: 43px;
        flex-basis: 43px;
        font-size: 20px;
    }

    .stat-value {
        font-size: 16px;
    }

    .summarygrid {
        grid-template-columns: 1fr 1fr;
    }
}

@media(max-width: 450px) {
    .stats {
        grid-template-columns: 1fr;
    }

    .brand h1 {
        font-size: 20px;
    }

    .summarygrid {
        grid-template-columns: 1fr 1fr;
    }

    .mini-number {
        font-size: 20px;
    }
}
</style>
</head>

<body>
<header class="topbar">
  <div class="brand">
    <div class="brandmark">♡</div>
    <div class="brandtext">
      <h1>ECG Insight</h1>
      <p>AI-Based ECG Anomaly Detection</p>
    </div>
  </div>

  <nav class="nav">
    <a class="active" href="/">⌂ &nbsp;Home</a>
    <a href="#about">ⓘ &nbsp;About</a>
    <a href="#how">♧ &nbsp;How It Works</a>
    <a href="#dataset">▤ &nbsp;Dataset</a>
  </nav>

  <div class="projecttag">Machine Learning Project</div>
</header>

<main>
  <section class="hero">
    <div>
      <h2>ECG Analysis Dashboard</h2>
      <p>
        Upload ECG signal segments and analyze them using a trained
        Random Forest model.<br>
        Detect normal or abnormal heartbeats with AI.
      </p>
    </div>

    <div class="hero-badge">
      Early Detection<br>
      Healthier Hearts<br>
      Brighter Futures
    </div>
  </section>

  <section class="stats">
    <div class="stat">
      <div class="stat-icon">▦</div>
      <div>
        <div class="stat-label">ML Model</div>
        <div class="stat-value">Random Forest</div>
        <div class="stat-note">Trained on MIT-BIH ECG segments</div>
      </div>
    </div>

    <div class="stat">
      <div class="stat-icon">▥</div>
      <div>
        <div class="stat-label">Input Length</div>
        <div class="stat-value">{{ sample_count }} samples</div>
        <div class="stat-note">Per ECG segment</div>
      </div>
    </div>

    <div class="stat">
      <div class="stat-icon">⚙</div>
      <div>
        <div class="stat-label">Decision Threshold</div>
        <div class="stat-value">{{ threshold }}</div>
        <div class="stat-note">Loaded from saved model</div>
      </div>
    </div>

    <div class="stat">
      <div class="stat-icon">▤</div>
      <div>
        <div class="stat-label">Supports</div>
        <div class="stat-value">Single or Multiple Segments</div>
        <div class="stat-note">One segment = {{ sample_count }} samples</div>
      </div>
    </div>
  </section>

  {% if error %}
    <div class="error">
      <strong>Unable to analyze the ECG data:</strong> {{ error }}
    </div>
  {% endif %}

  <section class="layout">
    <div class="leftcol">
      <div class="card">
        <div class="uploadhead">
          <div class="cardhead" style="padding:0">
            <div class="cardicon">♧</div>
            <div>
              <h3>Analyze ECG Signal</h3>
              <p>Upload a CSV file or enter ECG values to get predictions.</p>
            </div>
          </div>
        </div>

        <div class="uploadbody">
          <form method="POST" enctype="multipart/form-data">
            <div class="tabs">
              <div class="tab active">▱ &nbsp;Upload CSV File</div>
              <div class="tab">✎ &nbsp;Enter Values Manually</div>
            </div>

            <div class="filebox">
              <div class="cloud">☁</div>
              <strong>Choose a CSV file</strong>
              <p>
                Upload a file containing one or more ECG segments<br>
                (Each segment must have {{ sample_count }} numeric samples)
              </p>
              <input
                type="file"
                name="ecg_file"
                id="ecg_file"
                accept=".csv,.txt"
              >
            </div>

            <div class="info">
              <strong>ⓘ &nbsp;CSV Format Options:</strong>
              <ul>
                <li>One segment per row (recommended)</li>
                <li>One segment per column</li>
                <li>Or a single row/column with multiple of {{ sample_count }} values</li>
              </ul>
            </div>

            <label for="signal">Enter values manually (optional)</label>
            <textarea
              id="signal"
              name="signal"
              placeholder="Enter {{ sample_count }} numeric values separated by commas or spaces..."
            >{{ signal }}</textarea>

            <button type="submit">▶ &nbsp; Analyze ECG Signal</button>
          </form>
        </div>
      </div>

      <div class="card" id="about">
        <div class="cardhead">
          <div class="cardicon">▣</div>
          <div><h3>About This Project</h3></div>
        </div>

        <div class="aboutbody">
          <p>
            This academic project uses machine learning to classify ECG
            heartbeat segments as normal or abnormal using the MIT-BIH
            Arrhythmia dataset.
          </p>

          <div class="featuregrid">
            <div class="feature"><span>♧</span>Educational Prototype</div>
            <div class="feature"><span>♧</span>Machine Learning</div>
            <div class="feature"><span>⌁</span>ECG Signal Analysis</div>
            <div class="feature"><span>◎</span>Web Application</div>
          </div>

          <div class="warning">
            ⚠ &nbsp;<strong>Important:</strong>
            This model has not been clinically validated. Its output is
            experimental and must not be used to diagnose, rule out, or
            treat a heart condition.
          </div>
        </div>
      </div>
    </div>

    <div class="rightcol">
      <div class="card">
        <div class="cardhead">
          <div class="cardicon">⌁</div>
          <div>
            <h3>ECG Waveform (First Segment)</h3>
            <p>The normalized input signal is displayed below.</p>
          </div>
        </div>

        <div class="wavebody">
          {% if waveform %}
            <img
              src="data:image/png;base64,{{ waveform }}"
              alt="Normalized ECG waveform"
            >
          {% else %}
            <div class="emptychart">
              ECG waveform will appear here after you upload data and run analysis.
            </div>
          {% endif %}
        </div>
      </div>

      <div class="card">
        <div class="cardhead">
          <div class="cardicon">▥</div>
          <div>
            <h3>Prediction Summary</h3>
            <p>Overall results for all analyzed segments.</p>
          </div>
        </div>

        <div class="summarygrid">
          <div class="mini">
            <div class="mini-title">▤ &nbsp; Total Segments</div>
            <div class="mini-number">{{ summary.total if summary else 0 }}</div>
            <div class="mini-caption">Analyzed from input</div>
          </div>

          <div class="mini green">
            <div class="mini-title">✓ &nbsp; Normal Segments</div>
            <div class="mini-number">{{ summary.normal if summary else 0 }}</div>
            <div class="mini-caption">Predicted as normal</div>
          </div>

          <div class="mini red">
            <div class="mini-title">⚠ &nbsp; Abnormal Segments</div>
            <div class="mini-number">{{ summary.abnormal if summary else 0 }}</div>
            <div class="mini-caption">Flagged as abnormal</div>
          </div>

          <div class="mini pink">
            <div class="mini-title">◔ &nbsp; Average Abnormal Score</div>
            <div class="mini-number">
              {{ ((results|sum(attribute='score')) / (results|length))|round(2) if results else '0.00' }}%
            </div>
            <div class="mini-caption">Across all segments</div>
          </div>
        </div>
      </div>

      <div class="bottomgrid">
        <div class="card">
          <div class="cardhead">
            <div class="cardicon">◎</div>
            <div><h3>Prediction Results (First Segment)</h3></div>
          </div>

          <div class="resultbody">
            {% if results %}
              {% set first = results[0] %}

              <div class="predictionbox {{ 'abnormal' if first.abnormal else '' }}">
                <div class="predictiontitle">
                  {{ '⚠ Abnormal Pattern Flagged' if first.abnormal else '✓ Normal Pattern Predicted' }}
                </div>

                <div class="predictionnote">Model abnormal score</div>
                <div class="score">{{ first.score }}%</div>

                <div class="progress">
                  <div
                    class="progressinner"
                    style="width:{{ [first.score,100]|min }}%"
                  ></div>
                </div>

                <div class="predictionnote">
                  Decision threshold: {{ threshold }}. The model flags a
                  segment when its abnormal score reaches or exceeds this threshold.
                </div>
              </div>
            {% else %}
              <div class="emptychart" style="height:155px">
                Your first segment prediction will appear here after analysis.
              </div>
            {% endif %}
          </div>
        </div>

        <div class="card">
          <div class="cardhead">
            <div class="cardicon">▤</div>
            <div>
              <h3>All Segment Results</h3>
              <p>Prediction for each analyzed segment.</p>
            </div>
          </div>

          <div class="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Prediction</th>
                  <th>Abnormal Score</th>
                  <th>Result</th>
                </tr>
              </thead>

              <tbody>
                {% if results %}
                  {% for item in results %}
                    <tr>
                      <td>{{ item.index }}</td>
                      <td>{{ 'Abnormal' if item.abnormal else 'Normal' }}</td>
                      <td>{{ item.score }}%</td>
                      <td>
                        <span class="badge {{ 'bad' if item.abnormal else '' }}">
                          {{ 'Abnormal' if item.abnormal else 'Normal' }}
                        </span>
                      </td>
                    </tr>
                  {% endfor %}
                {% else %}
                  <tr>
                    <td colspan="4">Upload ECG data to view results.</td>
                  </tr>
                {% endif %}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>

<footer>
  ECG Insight · AI-Based ECG Anomaly Detection Using Machine Learning<br>
  Academic prototype only · Not intended for clinical diagnosis
</footer>
</body>
</html>
"""

# ==================================================
# 5. PARSE CSV INTO MULTIPLE SEGMENTS
# ==================================================

def read_segments(uploaded_file, manual_text):
    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename.lower()

        if not filename.endswith((".csv", ".txt")):
            raise ValueError("Please upload a .csv or .txt file.")

        try:
            content = uploaded_file.read().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "The file must be readable UTF-8 text."
            ) from exc

        if not content.strip():
            raise ValueError("The uploaded file is empty.")

        frame = pd.read_csv(
            StringIO(content),
            header=None,
            sep=r"[,;\s]+",
            engine="python",
            comment="#",
            skip_blank_lines=True,
        )

        numeric = frame.apply(pd.to_numeric, errors="coerce")
        raw_values = numeric.to_numpy(dtype=float)

        if np.isnan(raw_values).any():
            keep_rows = ~np.isnan(raw_values).all(axis=1)
            raw_values = raw_values[keep_rows]

            if raw_values.size:
                keep_cols = ~np.isnan(raw_values).all(axis=0)
                raw_values = raw_values[:, keep_cols]

        if raw_values.size == 0 or np.isnan(raw_values).any():
            raise ValueError(
                "The CSV contains non-numeric or missing cells. "
                "Upload a file containing only numeric ECG samples."
            )

        if not np.isfinite(raw_values).all():
            raise ValueError("The CSV contains invalid numeric values.")

        rows, cols = raw_values.shape

        if cols == EXPECTED_SAMPLES:
            segments = raw_values
        elif rows == EXPECTED_SAMPLES:
            segments = raw_values.T
        elif raw_values.size % EXPECTED_SAMPLES == 0:
            segments = raw_values.reshape(-1, EXPECTED_SAMPLES)
        else:
            raise ValueError(
                f"Could not split this CSV into segments of "
                f"{EXPECTED_SAMPLES} samples. Found {raw_values.size} "
                f"numeric values. The total must be a multiple of "
                f"{EXPECTED_SAMPLES}, or the CSV must have "
                f"{EXPECTED_SAMPLES} columns per segment."
            )

    elif manual_text.strip():
        values = np.fromstring(
            manual_text.replace(",", " ").replace(";", " "),
            sep=" ",
        )

        if len(values) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Manual input must contain exactly {EXPECTED_SAMPLES} "
                f"values; received {len(values)}."
            )

        segments = values.reshape(1, EXPECTED_SAMPLES)

    else:
        raise ValueError(
            "Upload a CSV file or enter ECG sample values."
        )

    if segments.ndim != 2 or segments.shape[1] != EXPECTED_SAMPLES:
        raise ValueError(
            f"Each ECG segment must contain exactly "
            f"{EXPECTED_SAMPLES} numeric samples."
        )

    if len(segments) > MAX_SEGMENTS:
        raise ValueError(
            f"This upload contains {len(segments)} segments. "
            f"The current limit is {MAX_SEGMENTS} segments per upload."
        )

    if not np.isfinite(segments).all():
        raise ValueError("The ECG data contains invalid numeric values.")

    for i, segment in enumerate(segments, start=1):
        if np.std(segment) <= 1e-8:
            raise ValueError(
                f"Segment {i} has almost no variation. "
                "Provide a valid ECG segment."
            )

    return segments.astype(np.float32)


# ==================================================
# 6. PREPROCESS AND PREDICT MULTIPLE SEGMENTS
# ==================================================

def prepare_features(segment):
    std = float(np.std(segment))

    if std <= 1e-8:
        raise ValueError("A segment has almost no variation.")

    normalized = (segment - np.mean(segment)) / std

    if FEATURE_ENGINEERING == "raw_waveform_plus_18_features":
        features = extract_features(normalized).reshape(1, -1)
    else:
        features = normalized.reshape(1, -1)

    expected_features = getattr(model, "n_features_in_", None)

    if (
        expected_features is not None
        and features.shape[1] != expected_features
    ):
        raise ValueError(
            f"The model expects {expected_features} features, but the app "
            f"generated {features.shape[1]}. Check the saved feature configuration."
        )

    return normalized, features


def predict_segments(segments):
    classes = [
        str(label).strip().lower()
        for label in model.classes_
    ]

    if "abnormal" not in classes:
        raise ValueError(
            f"Abnormal class not found. Model classes: {model.classes_}"
        )

    abnormal_index = classes.index("abnormal")
    results = []
    normalized_segments = []

    for i, segment in enumerate(segments, start=1):
        normalized, features = prepare_features(segment)
        probabilities = model.predict_proba(features)[0]

        abnormal_score = float(probabilities[abnormal_index])
        is_abnormal = abnormal_score >= THRESHOLD

        results.append({
            "index": i,
            "abnormal": is_abnormal,
            "prediction": (
                "Abnormal Pattern Flagged"
                if is_abnormal
                else "Normal Pattern Predicted"
            ),
            "score": round(abnormal_score * 100, 2),
        })

        normalized_segments.append(normalized)

    summary = {
        "total": len(results),
        "normal": sum(
            not item["abnormal"] for item in results
        ),
        "abnormal": sum(
            item["abnormal"] for item in results
        ),
    }

    return results, summary, normalized_segments

# ==================================================
# 7. MAIN ROUTE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    results = []
    summary = None
    waveform = None
    signal_text = ""

    if request.method == "POST":
        signal_text = request.form.get("signal", "").strip()

        try:
            uploaded_file = request.files.get("ecg_file")

            segments = read_segments(
                uploaded_file,
                signal_text
            )

            results, summary, normalized_segments = predict_segments(
                segments
            )

            waveform = create_waveform(
                normalized_segments[0]
            )

        except Exception as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        error=error,
        results=results,
        summary=summary,
        waveform=waveform,
        signal=signal_text,
        threshold=f"{THRESHOLD:.2f}",
        sample_count=EXPECTED_SAMPLES,
        max_segments=MAX_SEGMENTS,
    )


# ==================================================
# 8. RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    print("Starting ECG Insight Dashboard...")
    print("Open http://127.0.0.1:5000 in your browser.")

    app.run(
        debug=True,
        use_reloader=False
    )