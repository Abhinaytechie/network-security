# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from networksecurity.utils.main_utils.utils import load_object



# Constants
OUTPUT_PATH = "prediction_output/output.csv"
PREPROCESSOR_PATH = "final_model/preprocessor.pkl"
MODEL_PATH = "final_model/model.pkl"

# ─────────────────────────────
st.set_page_config(page_title="🔐 Network Security Suite", layout="wide")

st.title("🧠 Cybersecurity Threat Detection Suite")

# Main Navigation
main_mode = st.sidebar.radio("Choose Mode", ["🧪 Simple Mode", "🛠️ Advanced Mode"])

# ─────────────────────────────
if main_mode == "🧪 Simple Mode":
    menu = ["🏠 Home", "📁 Upload & Predict", "📊 Threat Analyzer"]
    simple_option = st.sidebar.radio("Select Module", menu)

    # ---- 🏠 Home ----
    if simple_option == "🏠 Home":
        st.markdown("### 📡 Network Security Analyzer")
        st.write("Welcome! This tool lets you upload network traffic data (CSV), predict threats, and visualize insights.")
        st.info("Start with **📁 Upload & Predict** or directly explore **📊 Threat Analyzer** with live test data.")

    # ---- 📁 Upload & Predict ----
    elif simple_option == "📁 Upload & Predict":
        uploaded_file = st.file_uploader("Upload network traffic CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview", df.head())

            if st.button("🔍 Predict Threats"):
                try:
                    model = load_object(MODEL_PATH)
                    preprocessor = load_object(PREPROCESSOR_PATH)

                    X = df.copy()
                    X_transformed = preprocessor.transform(X)
                    predictions = model.predict(X_transformed)

                    df["predicted_column"] = predictions
                    df.to_csv(OUTPUT_PATH, index=False)

                    st.success("✔️ Prediction completed. See Threat Analyzer for further details.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ---- 📊 Threat Analyzer ----
    elif simple_option == "📊 Threat Analyzer":
        st.header("🛡️ Advanced Threat Analyzer")

        uploaded_threat_file = st.file_uploader("Upload network traffic CSV for deep scan", type=["csv"])

        if uploaded_threat_file:
            df = pd.read_csv(uploaded_threat_file)
            st.write("✅ Uploaded Data Sample", df.head())

            try:
                model = load_object(MODEL_PATH)
                preprocessor = load_object(PREPROCESSOR_PATH)

                # Prepare data
                X_transformed = preprocessor.transform(df)
                predictions = model.predict(X_transformed)

                df["prediction"] = predictions

                # Show basic prediction stats
                total = len(df)
                threats = df[df["prediction"] == 1]
                st.markdown(f"### 🔍 Total Entries: {total} | 🚨 Threats Detected: {len(threats)}")

                if threats.empty:
                    st.success("✅ No threats detected.")
                else:
                    # ------------------ Feature 1: Auto Threat Tagging ------------------
                    def tag_threat(row):
                        if row['port'] in [21, 22, 23, 80, 443] and row['SSLfinal_State'] == 0:
                            return "Insecure Port Access"
                        elif row['web_traffic'] < 10000:
                            return "Low Traffic Anomaly"
                        elif row['Page_Rank'] < 0.2:
                            return "Suspicious Page Rank"
                        else:
                            return "Generic Threat"

                    threats["Threat_Tag"] = threats.apply(tag_threat, axis=1)

                    # ------------------ Feature 2: Summary ------------------
                    st.markdown("### 📊 Threat Summary")
                    tag_counts = threats["Threat_Tag"].value_counts()
                    for tag, count in tag_counts.items():
                        st.write(f"- **{tag}**: {count} cases")

                    # ------------------ Feature 3: Top Risk Patterns ------------------
                    st.markdown("### 🔁 Top Threat Patterns")
                    common_patterns = threats[["having_IP_Address", "port", "Page_Rank"]].value_counts().head(5)
                    st.dataframe(common_patterns.rename("Count").reset_index())

                    # ------------------ Feature 4: Anomaly Detection ------------------
                    from sklearn.ensemble import IsolationForest

                    iso = IsolationForest(contamination=0.05, random_state=42)
                    anomaly_preds = iso.fit_predict(X_transformed)

                    df["anomaly_score"] = anomaly_preds
                    anomalies = df[df["anomaly_score"] == -1]

                    st.markdown("### 🧬 Anomaly Detection")
                    if anomalies.empty:
                        st.success("No anomalies detected outside the model's knowledge.")
                    else:
                        st.warning(f"⚠️ {len(anomalies)} anomalous entries found using IsolationForest!")
                        st.dataframe(anomalies.head())

                    # ------------------ Download ------------------
                    st.download_button("📥 Download Threat Results", data=threats.to_csv(index=False), file_name="threat_results.csv")

            except Exception as e:
                st.error(f"❌ Threat analysis failed: {e}")


# ─────────────────────────────
elif main_mode == "🛠️ Advanced Mode":
    option = st.sidebar.radio("Go to", ["🏠 Dashboard", "📊 Feature Insights", "📄 Export Summary"])

    # ---- Dashboard ----
    if option == "🏠 Dashboard":
        st.title("📊 Network Security Dashboard")

        if os.path.exists(OUTPUT_PATH):
            df = pd.read_csv(OUTPUT_PATH)

            st.markdown("### 🧾 Summary")
            st.write(f"**Total Rows:** {df.shape[0]}")
            st.write(f"**Classes Found:** {df['predicted_column'].nunique()}")
            st.dataframe(df.head())

            st.markdown("### 📌 Class Distribution")
            count_df = df['predicted_column'].value_counts().reset_index()
            count_df.columns = ['label', 'count']
            fig = px.bar(count_df, x='label', y='count', color='label', title='Prediction Class Distribution', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🥧 Class Share")
            fig2 = px.pie(df, names='predicted_column', title='Class-wise Share')
            st.plotly_chart(fig2, use_container_width=True)

            if 'actual_label' in df.columns:
                st.markdown("### 🔥 Confusion Matrix")
                cm = pd.crosstab(df['actual_label'], df['predicted_column'])
                fig3, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig3)

        else:
            st.info("No predictions found. Upload a file in the 📤 Predict section.")

    

    # ---- Feature Insights ----
    elif option == "📊 Feature Insights":
        st.title("📊 Feature Insights & Data Understanding")

        if os.path.exists(OUTPUT_PATH):
            df = pd.read_csv(OUTPUT_PATH)
            if "predicted_column" in df.columns:
                features_df = df.drop(columns=["predicted_column"])
            else:
                features_df = df

            st.subheader("📋 Basic Data Summary")
            st.dataframe(features_df.describe().T)

            st.subheader("🧮 Interactive Correlation Heatmap")
            corr = features_df.corr(numeric_only=True)
            top_n = st.slider("Select Top N Correlated Features to Display", min_value=5, max_value=min(30, len(corr)), value=10)

            corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
            corr_unstacked = corr_unstacked[corr_unstacked < 1]
            top_features = set()
            for i in range(top_n):
                top_features.update(corr_unstacked.index[i])
            corr_top = corr.loc[list(top_features), list(top_features)]

            fig_corr = px.imshow(corr_top, text_auto=".2f", color_continuous_scale="RdBu_r", title="Top-N Correlated Features", aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

            if "predicted_column" in df.columns:
                st.subheader("📡 Class-wise Feature Radar Plot")
                class_means = df.groupby("predicted_column").mean(numeric_only=True)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(class_means)
                radar_df = pd.DataFrame(scaled, index=class_means.index, columns=class_means.columns)

                fig_radar = go.Figure()
                for idx in radar_df.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_df.loc[idx].values,
                        theta=radar_df.columns,
                        fill='toself',
                        name=f"Class {idx}"
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="📊 Class Distribution Radar"
                )
                st.plotly_chart(fig_radar)
        else:
            st.warning("⚠️ No prediction data found.")

    # ---- Export Summary ----
    # ---- Export Summary ----
elif option == "📄 Export Summary":
    st.title("📄 Export Prediction Summary")

    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        summary = {
            "Total Rows": df.shape[0],
            "Predicted Classes": df['predicted_column'].nunique(),
            "Top Class": df['predicted_column'].value_counts().idxmax(),
            "Class Distribution": df['predicted_column'].value_counts().to_dict()
        }

        st.json(summary)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="predictions.csv")

        # ---------------- PDF Export Option ----------------
        st.markdown("---")
        st.subheader("🖨️ Generate PDF Report")
        import pdfkit
        from tempfile import NamedTemporaryFile

        html_content = f"""
        <h1>Cybersecurity Threat Report</h1>
        <p><b>Total Rows:</b> {summary['Total Rows']}</p>
        <p><b>Predicted Classes:</b> {summary['Predicted Classes']}</p>
        <p><b>Top Class:</b> {summary['Top Class']}</p>
        <h3>Class Distribution:</h3>
        <ul>
        {''.join([f"<li>{label}: {count}</li>" for label, count in summary['Class Distribution'].items()])}
        </ul>
        <p><i>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i></p>
        """

        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                pdfkit.from_string(html_content, tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as f:
                    st.download_button("📄 Download PDF Report", data=f, file_name="threat_summary.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    else:
        st.info("No predictions available.")

