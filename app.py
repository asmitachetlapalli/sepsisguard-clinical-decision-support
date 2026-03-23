#!/usr/bin/env python3
"""
SepsisGuard - Clinical Decision Support Dashboard
Streamlit app combining XGBoost + LR prediction with RAG-based recommendations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

st.set_page_config(page_title="SepsisGuard", page_icon="🏥", layout="wide")


# ── Model Loading ───────────────────────────────────────────────────────
@st.cache_resource
def load_xgb_model():
    import joblib
    path = PROJECT_ROOT / "models" / "xgboost_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_lr_model():
    import joblib
    path = PROJECT_ROOT / "models" / "baseline_lr.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_rag_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        chroma_dir = PROJECT_ROOT / "rag" / "chroma_db"
        if not chroma_dir.exists():
            return None
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_collection(name="sepsis_guidelines", embedding_function=ef)
    except Exception:
        return None


# ── Prediction ──────────────────────────────────────────────────────────
def predict_xgb(vitals: dict, model_data: dict) -> float:
    feature_cols = model_data["feature_cols"]
    X = np.array([[vitals.get(c, 0) for c in feature_cols]])
    return float(model_data["model"].predict_proba(X)[0][1])


def predict_lr(vitals: dict, model_data: dict) -> float:
    feature_cols = model_data["feature_cols"]
    X = np.array([[vitals.get(c, 0) for c in feature_cols]])
    return float(model_data["model"].predict_proba(X)[0][1])


def get_rag_recommendation(patient_data: dict, risk_score: float) -> str | None:
    api_key = os.environ.get("GOOGLE_API_KEY") or st.session_state.get("api_key")
    if not api_key:
        return None
    try:
        from rag_engine import generate_recommendation
        return generate_recommendation(patient_data, risk_score, api_key=api_key)
    except Exception as e:
        return f"Error generating recommendation: {e}"


# ── UI ──────────────────────────────────────────────────────────────────
def main():
    st.title("SepsisGuard: Clinical Decision Support")
    st.caption("AI-powered early sepsis detection with explainable recommendations")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google Gemini API Key (free)",
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey",
        )
        if api_key:
            st.session_state["api_key"] = api_key

        st.divider()
        st.header("Model Status")

        xgb_data = load_xgb_model()
        lr_data = load_lr_model()
        rag_collection = load_rag_collection()

        if xgb_data:
            st.success(f"XGBoost loaded (AUROC: {xgb_data.get('auroc', 0):.4f})")
        else:
            st.warning("XGBoost not found — run: python models/train_xgboost.py")

        if lr_data:
            st.success("Logistic Regression loaded (AUROC: 0.7074)")
        else:
            st.warning("LR not found — run: python models/baseline_lr.py")

        if rag_collection:
            st.success(f"RAG: {rag_collection.count()} guideline chunks")
        else:
            st.warning("RAG not built — run: python rag/build_vectordb.py")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Patient Assessment", "Batch Analysis", "About"])

    with tab1:
        st.header("Patient Vital Signs")
        col1, col2, col3 = st.columns(3)

        with col1:
            hr = st.number_input("Heart Rate (bpm)", 30, 250, 85)
            o2sat = st.number_input("SpO2 (%)", 50, 100, 97)
            temp = st.number_input("Temperature (°C)", 33.0, 42.0, 37.0, step=0.1)
            sbp = st.number_input("Systolic BP (mmHg)", 40, 250, 120)

        with col2:
            dbp = st.number_input("Diastolic BP (mmHg)", 20, 150, 70)
            map_val = st.number_input("MAP (mmHg)", 20, 200, 75)
            resp = st.number_input("Respiratory Rate", 5, 60, 16)
            age = st.number_input("Age (years)", 18, 100, 55)

        with col3:
            lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 1.0, step=0.1)
            wbc = st.number_input("WBC (x10³/μL)", 0.0, 50.0, 8.0, step=0.5)
            creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, step=0.1)
            iculos = st.number_input("ICU Length of Stay (hours)", 1, 500, 24)

        patient_data = {
            "HR": hr, "O2Sat": o2sat, "Temp": temp, "SBP": sbp,
            "DBP": dbp, "MAP": map_val, "Resp": resp, "Age": age,
            "Lactate": lactate, "WBC": wbc, "Creatinine": creatinine,
            "ICULOS": iculos, "Gender": 0, "HospAdmTime": 0,
            "Bilirubin_total": 0, "Platelets": 200, "BUN": 15,
            "Glucose": 100, "Hgb": 12, "Hct": 36, "pH": 7.4,
        }

        if st.button("Assess Sepsis Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                risk_scores = {}

                if xgb_data:
                    risk_scores["XGBoost"] = predict_xgb(patient_data, xgb_data)
                if lr_data:
                    risk_scores["Logistic Regression"] = predict_lr(patient_data, lr_data)

                if not risk_scores:
                    st.error("No models available. Train models first.")
                    return

                # Primary risk = XGBoost if available, else LR
                primary_name = "XGBoost" if "XGBoost" in risk_scores else "Logistic Regression"
                primary_risk = risk_scores[primary_name]

                # Display
                st.subheader("Risk Assessment")
                color = "🔴" if primary_risk >= 0.7 else "🟡" if primary_risk >= 0.4 else "🟢"
                level = "HIGH" if primary_risk >= 0.7 else "MODERATE" if primary_risk >= 0.4 else "LOW"

                risk_cols = st.columns(len(risk_scores))
                for i, (name, score) in enumerate(risk_scores.items()):
                    with risk_cols[i]:
                        c = "🔴" if score >= 0.7 else "🟡" if score >= 0.4 else "🟢"
                        lv = "HIGH" if score >= 0.7 else "MODERATE" if score >= 0.4 else "LOW"
                        label = f"{name} {'(Primary)' if name == primary_name else '(Baseline)'}"
                        st.metric(label, f"{c} {score:.1%}", lv)

                # Vital sign flags
                st.subheader("Vital Sign Flags")
                flags = []
                if hr > 90: flags.append(("Heart Rate", f"{hr} bpm", "Tachycardia (>90)"))
                if temp > 38.3: flags.append(("Temperature", f"{temp}°C", "Fever (>38.3)"))
                if temp < 36: flags.append(("Temperature", f"{temp}°C", "Hypothermia (<36)"))
                if map_val < 65: flags.append(("MAP", f"{map_val} mmHg", "Hypotension (<65)"))
                if resp > 22: flags.append(("Resp Rate", f"{resp}/min", "Tachypnea (>22)"))
                if o2sat < 92: flags.append(("SpO2", f"{o2sat}%", "Hypoxemia (<92)"))
                if lactate > 2: flags.append(("Lactate", f"{lactate} mmol/L", "Elevated (>2)"))
                if wbc > 12: flags.append(("WBC", f"{wbc} x10³/μL", "Leukocytosis (>12)"))
                if creatinine > 2: flags.append(("Creatinine", f"{creatinine} mg/dL", "Renal concern (>2)"))

                if flags:
                    flag_df = pd.DataFrame(flags, columns=["Vital Sign", "Value", "Flag"])
                    st.dataframe(flag_df, use_container_width=True, hide_index=True)
                else:
                    st.success("All vital signs within normal ranges.")

                # RAG Recommendation
                st.subheader("Clinical Recommendation (RAG)")
                if rag_collection and (api_key or os.environ.get("GOOGLE_API_KEY")):
                    with st.spinner("Generating evidence-based recommendation..."):
                        rec = get_rag_recommendation(patient_data, primary_risk)
                        if rec:
                            st.markdown(rec)
                        else:
                            st.info("Could not generate recommendation.")
                elif rag_collection:
                    st.info("Add a Gemini API key in the sidebar for AI recommendations. Showing relevant guidelines:")
                    query = f"sepsis management risk {primary_risk:.0%}"
                    results = rag_collection.query(query_texts=[query], n_results=3)
                    for doc in results["documents"][0]:
                        st.markdown(f"> {doc}")
                else:
                    st.info("Build the RAG database to enable recommendations (`python rag/build_vectordb.py`)")

                st.divider()
                st.caption("⚠️ AI-generated clinical decision support. All recommendations require physician review.")

    with tab2:
        st.header("Batch Patient Analysis")
        st.write("Upload a CSV with patient vitals to assess multiple patients.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Assessment"):
                model_data = xgb_data or lr_data
                if model_data:
                    feature_cols = model_data["feature_cols"]
                    X = batch_df.reindex(columns=feature_cols, fill_value=0).values
                    probs = model_data["model"].predict_proba(X)[:, 1]
                    batch_df["risk_score"] = probs
                    batch_df["risk_level"] = pd.cut(
                        probs, bins=[0, 0.4, 0.7, 1.0],
                        labels=["LOW", "MODERATE", "HIGH"]
                    )
                    st.dataframe(batch_df, use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Patients", len(batch_df))
                    c2.metric("High Risk", (batch_df["risk_level"] == "HIGH").sum())
                    c3.metric("Moderate Risk", (batch_df["risk_level"] == "MODERATE").sum())

    with tab3:
        st.header("About SepsisGuard")
        st.markdown("""
        **SepsisGuard** is an AI-powered clinical decision support system for early sepsis detection in ICU settings.

        ### Models
        | Model | Features | AUROC | Role |
        |-------|----------|-------|------|
        | **XGBoost** | 21 features (vitals + labs + demographics) | **0.79** | Primary |
        | **Logistic Regression** | 6 features (vitals only) | **0.71** | Baseline |

        ### RAG Pipeline
        Retrieves relevant Surviving Sepsis Campaign 2021 guidelines and generates
        explainable, evidence-based recommendations using Google Gemini (free tier).

        ### Data
        Trained on PhysioNet Challenge 2019 data (1,000 ICU patients, 38,809 hourly records)

        ### Disclaimer
        This tool is for educational and research purposes only. NOT a substitute for
        professional medical judgment.
        """)


if __name__ == "__main__":
    main()
