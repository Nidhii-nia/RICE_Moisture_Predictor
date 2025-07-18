# 🌾 RICE_Moisture_Predictor

Rice Moisture Predictor is a Streamlit web application for analyzing the moisture content of rice samples using RF measurement data (8–12 GHz), with automated grading, storage recommendations, and comparison/analytics of multiple runs.

---

## 🚀 Features

- 📁 Upload RF measurement CSV files (VNA data: S11/S21 in 8–12 GHz)
- 🤖 Predicts rice moisture content using a trained model
- 🧪 Displays permittivity calculations and grading table (IRRI-style, detailed storage & safety)
- 🔄 Remembers past session results and enables multi-sample comparison
- 📊 Visualizes key permittivity and moisture features interactively

---

## 🟢 Getting Started

### Option 1: Online (Recommended)

- Hosted on Streamlit Community Cloud (replace with your app link)
- No installation required—just upload your RF data CSV

### Option 2: Local Installation

1. Clone this repo:

<pre><code>```bash
git clone https://github.com/yourusername/rice-moisture-analyzer.git
cd rice-moisture-analyzer
```</code></pre>
2. Install requirements:

<pre><code>```bash
pip install -r requirements.txt
```</code></pre>


3. Ensure `moisture_model.pkl` is present in the app directory.

4. Run the app locally:
<pre><code>
```bash
streamlit run app.py
```</code></pre>

---

## 📖 How to Use

- Upload a compatible CSV (check "CSV Format Requirements" in app for details).
- Select desired analysis settings and frequencies.
- Click "Calculate" to analyze permittivity and predict moisture content.
- Explore the grading table, prediction, and compare multiple analyzed samples in your session.

---

## 📁 File Format Requirements

Your CSV must include the following columns:

<pre><code>
```text
freq[Hz], re:S11, im:S11, re:S21, im:S21
```</code></pre>

Typical frequency range: 8–12 GHz

---

## 🔬 Science & Model

- Uses permittivity features (ε′, ε″, |ε|) from VNA measurements
- Predicts moisture content via a machine learning model (`moisture_model.pkl`)
- Grading and storage guidance based on IRRI and recent research

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:

<pre><code>
```text
streamlit
pandas
numpy
plotly
joblib
```</code></pre>

---

## 👤 Credits

- Developed by [Nidhi]
- Research support: [CSIR-CSIO Chandigarh]
