import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from datetime import datetime

# ================== CONSTANTS ==================
L = 0.0045          # Sample thickness (m)
lamdac = 0.04572    # Cutoff wavelength (m)
c = 299792458       # Speed of light (m/s)
MIN_FREQ = 8.0      # Minimum frequency in GHz
MAX_FREQ = 12.0     # Maximum frequency in GHz

# --- Moisture Content Grading Table (Generalized, detailed) ---
GRADING_TABLE = {
    "Moisture Content (%)": ["≤13.0", "13.1–14.0", "14.1–15.0", "15.1–16.0", ">16.0"],
    "Storage Period": [
        "1 year+ (optimal)",
        "Up to 6–12 months",
        "Up to 3–4 months",
        "<2 months",
        "<1–2 weeks (unsafe)"
    ],
    "Quality & Aging": [
        "Minimal quality loss; good eating quality and aroma retained; slow increase in rancidity and hardness.",
        "Acceptable; slight increase in fatty acid value and hardness possible after several months.",
        "Noticeable quality decline (texture, taste); increased risk of aging and off-flavors.",
        "Rapid quality loss; substantial increase in free fatty acids, musty odor, harder rice.",
        "Severe and rapid deterioration; unacceptable sensory and nutritional quality."
    ],
    "Health/Safety": [
        "Low risk for mold, insects, or spoilage; safe for long-term storage.",
        "Low to moderate risk of spoilage; still safe with good ventilation/storage.",
        "Growing risk of mold and insect infestation, especially at higher temperatures.",
        "High risk of spoilage, unsafe for human consumption after short storage.",
        "Very high risk of mold, mycotoxins, and pathogenic bacteria; not safe."
    ],
    "Recommended Use": [
        "Ideal for long-term storage",
        "General market storage/sale",
        "Short-term storage",
        "Only for rapid turnover",
        "Not recommended; urgent drying"
    ]
}

# --- Initialize session state for history, once, per session ---
if 'run_history' not in st.session_state:
    st.session_state['run_history'] = []

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('moisture_model.pkl')
        return model
    except Exception as e:
        st.error(f"""**Model Error**: {str(e)}
                Please ensure 'moisture_model.pkl' exists in the app folder""")
        st.stop()

model = load_model()

# ================== MAIN APP ==================
def main():
    st.set_page_config(page_title="Rice Moisture Analyzer (8-12GHz)", layout="wide")
    st.title(f" Rice Moisture Predictor ({MIN_FREQ}-{MAX_FREQ}GHz)")
    
    # --- File Upload ---
    with st.expander("📋 CSV Format Requirements (8-12GHz range)", expanded=False):
        st.markdown(f"""
        **Required columns (case-sensitive):**
        - `freq[Hz]`: Frequency in Hertz (8-12GHz range, e.g. 8.2e9)
        - `re:S11`, `im:S11`: Real/imaginary parts of S11
        - `re:S21`, `im:S21`: Real/imaginary parts of S21
        """)
        
        sample_csv = pd.DataFrame({
            'freq[Hz]': [8.2e9, 10.0e9, 12.0e9],
            're:S11': [0.523, 0.481, 0.449],
            'im:S11': [-0.102, -0.085, -0.072],
            're:S21': [0.851, 0.812, 0.783],
            'im:S21': [0.312, 0.298, 0.281]
        }).to_csv(index=False)
        
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv,
            file_name="sample_vna_data.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(f"**Step 1:** Upload VNA Data (CSV, {MIN_FREQ}-{MAX_FREQ}GHz)", type="csv")
    
    # --- Frequency Selection ---
    st.subheader("**Step 2:** Analysis Settings")
    freq_option = st.radio(
        "Frequency Mode:",
        [f"Full Sweep ({MIN_FREQ}-{MAX_FREQ}GHz)", 
         "Specific Frequencies", 
         "Single Frequency"],
        horizontal=True
    )
    
    target_freqs = []
    if freq_option == "Specific Frequencies":
        freq_input = st.text_input(f"Enter frequencies (GHz) between {MIN_FREQ}-{MAX_FREQ}, comma-separated", "8.2,10.0,12.0")
        target_freqs = [float(f.strip()) for f in freq_input.split(",") if f.strip()]
    elif freq_option == "Single Frequency":
        target_freqs = [st.number_input(f"Enter frequency (GHz)", min_value=MIN_FREQ, max_value=MAX_FREQ, value=10.0)]

    run_label = None
    if uploaded_file:
        run_label = uploaded_file.name
    else:
        run_label = "No Name"

    if st.button("**Calculate Permittivity & Predict Moisture**", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a CSV file first!")
        else:
            with st.spinner("Analyzing..."):
                process_data(uploaded_file, target_freqs if freq_option != f"Full Sweep ({MIN_FREQ}-{MAX_FREQ}GHz)" else None, run_label)

    st.divider()
    st.subheader("📂 Previously Analyzed Results (this session)")

    display_history_and_plot()

# --- Data Processing ---
def process_data(uploaded_file, target_freqs=None, run_label=None):
    try:
        data = pd.read_csv(uploaded_file)
        required_cols = ['freq[Hz]', 're:S11', 'im:S11', 're:S21', 'im:S21']
        if not all(col in data.columns for col in required_cols):
            st.error("CSV missing required columns! Check the format guidelines.")
            return

        freqGHz = data['freq[Hz]'] / 1e9
        
        # Validate frequency range
        if (freqGHz.min() < MIN_FREQ) or (freqGHz.max() > MAX_FREQ):
            st.error(f"Data must be in {MIN_FREQ}-{MAX_FREQ} GHz range! Your data: {freqGHz.min():.2f}-{freqGHz.max():.2f} GHz")
            return

        if target_freqs:
            invalid_freqs = [f for f in target_freqs if f < MIN_FREQ or f > MAX_FREQ]
            if invalid_freqs:
                st.error(f"Invalid frequencies: {invalid_freqs}. Must be {MIN_FREQ}-{MAX_FREQ} GHz!")
                return

        s11 = data['re:S11'] + 1j * data['im:S11']
        s21 = data['re:S21'] + 1j * data['im:S21']

        # --- Permittivity Calculation ---
        lamda0 = c / (freqGHz * 1e9)
        X = ((s11**2 - s21**2) + 1) / (2 * s11 + 1e-12)
        Refcof1 = X + np.sqrt(X**2 - 1 + 0j)
        Ur = 1
        term1 = Ur * ((1 - Refcof1)**2 / (1 + Refcof1)**2) * (1 - (lamda0**2/lamdac**2))
        term2 = (lamda0**2/lamdac**2) * (1/(Ur + 1e-12))
        Er = term1 + term2

        # --- Prepare Results ---
        if target_freqs:
            indices = [np.abs(freqGHz - freq).argmin() for freq in target_freqs]
            results = pd.DataFrame({
                'Frequency (GHz)': freqGHz.iloc[indices],
                'ε\' (Real)': np.real(Er[indices]),
                'ε\'\' (Imaginary)': np.imag(Er[indices]),
                '|ε| (Magnitude)': np.abs(Er[indices])
            })
            avg_real = results['ε\' (Real)'].mean()
            avg_imag = results['ε\'\' (Imaginary)'].mean()
            avg_mag = results['|ε| (Magnitude)'].mean()
        else:
            results = pd.DataFrame({
                'Frequency (GHz)': freqGHz,
                'ε\' (Real)': np.real(Er),
                'ε\'\' (Imaginary)': np.imag(Er),
                '|ε| (Magnitude)': np.abs(Er)
            })
            avg_real = results['ε\' (Real)'].mean()
            avg_imag = results['ε\'\' (Imaginary)'].mean()
            avg_mag = results['|ε| (Magnitude)'].mean()

        # --- Predict Moisture
        moisture = model.predict([[avg_real, avg_imag, avg_mag]])[0]
        grade_row = assign_grade_row(moisture)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # --- Store result in session history
        st.session_state['run_history'].append({
            "timestamp": now_str,
            "file": run_label,
            "moisture": moisture,
            "avg_real": avg_real,
            "avg_imag": avg_imag,
            "avg_mag": avg_mag,
            "grading_category": grade_row["Moisture Content (%)"]
        })

        # --- Display Results ---
        st.success("Analysis Complete!")
        
        # 1. Show Permittivity Values
        st.subheader("📊 Permittivity Results")
        if target_freqs:
            st.dataframe(results.style.format({
                'Frequency (GHz)': '{:.2f}',
                'ε\' (Real)': '{:.4f}',
                'ε\'\' (Imaginary)': '{:.4f}',
                '|ε| (Magnitude)': '{:.4f}'
            }))
        else:
            st.write(f"**Average Values Across {MIN_FREQ}-{MAX_FREQ}GHz Range:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("ε\' (Real)", f"{avg_real:.4f}")
            col2.metric("ε\'\' (Imaginary)", f"{avg_imag:.4f}")
            col3.metric("|ε| (Magnitude)", f"{avg_mag:.4f}")

        # 2. Moisture Prediction
        st.subheader("💧 Moisture Prediction")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Moisture Content", f"{moisture:.2f}%")
        col2.metric("Moisture Grading Category", grade_row['Moisture Content (%)'])

        # 3. Interactive Plot - for current run only
        st.subheader("📈 Frequency Response (Current Run)")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if target_freqs:
            fig.add_trace(go.Scatter(
                x=results['Frequency (GHz)'], y=results['ε\' (Real)'],
                name="ε' (Real)", mode='markers+lines', marker=dict(color='red', size=10)),
                secondary_y=False
            )
            fig.add_trace(go.Scatter(
                x=results['Frequency (GHz)'], y=results['ε\'\' (Imaginary)'],
                name="ε'' (Imaginary)", mode='markers+lines', marker=dict(color='blue', size=10)),
                secondary_y=False
            )
        else:
            fig.add_trace(go.Scatter(
                x=freqGHz, y=np.real(Er), name="ε' (Real)", line=dict(color='red')),
                secondary_y=False
            )
            fig.add_trace(go.Scatter(
                x=freqGHz, y=np.imag(Er), name="ε'' (Imaginary)", line=dict(color='blue')),
                secondary_y=False
            )
        
        fig.add_hline(
            y=moisture, line=dict(color='green', dash='dash'),
            annotation_text=f"Moisture: {moisture:.2f}%", 
            annotation_position="bottom right",
            secondary_y=True
        )
        
        fig.update_layout(
            title=f"Permittivity vs Frequency ({MIN_FREQ}-{MAX_FREQ}GHz)",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Permittivity Value",
            yaxis2_title="Moisture Content (%)",
            height=500,
            xaxis_range=[MIN_FREQ, MAX_FREQ]
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Detailed Moisture Content Grading Table
        st.subheader("🌾 Rice Moisture Content Grading Table")
        grading_df = pd.DataFrame(GRADING_TABLE)
        # Highlight matching row
        def highlight_row(row):
            return ['background-color: #e5ffe5' if row.name == grade_row['index'] else '' for _ in row]
        st.dataframe(grading_df.style.apply(highlight_row, axis=1))

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

def assign_grade_row(moisture):
    # Returns a dict for the full row in the grading table
    index = None
    if moisture <= 13.0:
        index = 0
    elif 13.0 < moisture <= 14.0:
        index = 1
    elif 14.0 < moisture <= 15.0:
        index = 2
    elif 15.0 < moisture <= 16.0:
        index = 3
    else:
        index = 4
    row = {col: GRADING_TABLE[col][index] for col in GRADING_TABLE}
    row["index"] = index
    return row

def display_history_and_plot():
    run_history = st.session_state['run_history']
    if len(run_history) == 0:
        st.info("No previous results stored for this session yet.")
        return

    history_df = pd.DataFrame(run_history)
    st.dataframe(
        history_df[["timestamp", "file", "moisture", "grading_category"]]
        .rename(columns={"file": "File", "moisture": "Moisture (%)", "grading_category": "Grading Category"})
    )

    # Select samples for comparison plot
    st.subheader("🔍 Compare Multiple Analyses")
    indices = st.multiselect(
        "Select previously analyzed runs to compare on plot below (in order)",
        options=list(range(len(history_df))),
        default=list(range(len(history_df))),
        format_func=lambda idx: f"{history_df.iloc[idx]['file']} @ {history_df.iloc[idx]['timestamp']}"
    )

    if indices:
        plot_df = history_df.iloc[indices]
        fig = go.Figure()
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Bar(
                name=f"{row['file']} [{row['grading_category']}]",
                x=['ε\' (Real)', 'ε\'\' (Imaginary)', 'Moisture'],
                y=[row['avg_real'], row['avg_imag'], row['moisture']]
            ))
        fig.update_layout(
            barmode='group',
            title="Comparison: Permittivity and Predicted Moisture",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
