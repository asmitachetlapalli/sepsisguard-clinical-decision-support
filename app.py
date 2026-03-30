#!/usr/bin/env python3
"""SepsisGuard - Clinical Decision Support Dashboard"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Fallback .env loading
if not os.environ.get("GOOGLE_API_KEY"):
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()

st.set_page_config(page_title="SepsisGuard", page_icon="🏥", layout="wide")


# ── Load models once ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    # LSTM (primary)
    lstm_path = PROJECT_ROOT / "models" / "lstm_trained.pth"
    if lstm_path.exists():
        from lstm_model import SepsisLSTM
        ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
        lstm = SepsisLSTM(
            input_size=len(ckpt["feature_cols"]),
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        lstm.load_state_dict(ckpt["model_state_dict"])
        lstm.eval()
        models["lstm"] = {"model": lstm, **ckpt}

    # XGBoost
    xgb_path = PROJECT_ROOT / "models" / "xgboost_model.pkl"
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)

    # LR (baseline)
    lr_path = PROJECT_ROOT / "models" / "baseline_lr.pkl"
    if lr_path.exists():
        models["lr"] = joblib.load(lr_path)
    return models


@st.cache_resource
def load_rag():
    """Load ChromaDB collection and sentence-transformer model separately.
    Returns (collection_without_ef, encode_function) to avoid deadlock."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        chroma_dir = PROJECT_ROOT / "rag" / "chroma_db"
        if not chroma_dir.exists():
            return None, None
        # Load embedding model separately so we control when encoding happens
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load collection WITHOUT embedding function — we'll pass embeddings manually
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("sepsis_guidelines")
        return collection, embed_model
    except Exception:
        return None, None


models = load_models()
rag_collection, embed_model = load_rag()
rag = rag_collection  # for sidebar status check
api_key = os.environ.get("GOOGLE_API_KEY")


# ── Main UI ─────────────────────────────────────────────────────────────
st.title("SepsisGuard: Clinical Decision Support")
st.caption("AI-powered early sepsis detection with explainable recommendations")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    sidebar_key = st.text_input("Google Gemini API Key", type="password",
                                 help="Or set GOOGLE_API_KEY in .env")
    if sidebar_key:
        api_key = sidebar_key

    st.divider()
    st.header("Model Status")
    if "lstm" in models:
        st.success(f"LSTM loaded (AUROC: {models['lstm'].get('best_auroc', 0):.4f}) — Primary")
    else:
        st.warning("LSTM not found")
    if "xgb" in models:
        st.success(f"XGBoost loaded (AUROC: {models['xgb'].get('auroc', 0):.4f})")
    else:
        st.warning("XGBoost not found")
    if "lr" in models:
        st.success("LR loaded — Baseline")
    else:
        st.warning("LR not found")
    if rag:
        st.success(f"RAG: {rag.count()} guideline chunks")
    else:
        st.warning("RAG not built")
    if api_key:
        st.success("Gemini API key loaded")
    else:
        st.warning("No Gemini API key")

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

    patient = {
        "HR": hr, "O2Sat": o2sat, "Temp": temp, "SBP": sbp,
        "DBP": dbp, "MAP": map_val, "Resp": resp, "Age": age,
        "Lactate": lactate, "WBC": wbc, "Creatinine": creatinine,
        "ICULOS": iculos, "Gender": 0, "HospAdmTime": 0,
        "Bilirubin_total": 0, "Platelets": 200, "BUN": 15,
        "Glucose": 100, "Hgb": 12, "Hct": 36, "pH": 7.4,
    }

    if st.button("Assess Sepsis Risk", type="primary", use_container_width=True):

        # ── Risk Scores ─────────────────────────────────────────────
        scores = {}

        # LSTM (primary)
        if "lstm" in models:
            lstm_data = models["lstm"]
            feature_cols = lstm_data["feature_cols"]
            scaler_mean = np.array(lstm_data["scaler_mean"])
            scaler_scale = np.array(lstm_data["scaler_scale"])
            seq_len = lstm_data["seq_len"]

            raw = np.array([patient.get(c, 0) for c in feature_cols], dtype=np.float32)
            normalized = (raw - scaler_mean) / scaler_scale
            seq = np.tile(normalized, (seq_len, 1))
            tensor = torch.FloatTensor(seq).unsqueeze(0)
            with torch.no_grad():
                scores["LSTM"] = float(lstm_data["model"](tensor, apply_sigmoid=True).item())

        # XGBoost
        if "xgb" in models:
            xgb_data = models["xgb"]
            X = np.array([[patient.get(c, 0) for c in xgb_data["feature_cols"]]])
            scores["XGBoost"] = float(xgb_data["model"].predict_proba(X)[0][1])

        # LR (baseline)
        if "lr" in models:
            lr_data = models["lr"]
            X = np.array([[patient.get(c, 0) for c in lr_data["feature_cols"]]])
            scores["Logistic Regression"] = float(lr_data["model"].predict_proba(X)[0][1])

        if not scores:
            st.error("No models available.")
        else:
            primary = list(scores.keys())[0]  # LSTM first if available
            primary_risk = scores[primary]

            # Get risk thresholds from XGBoost model (or use defaults)
            if "xgb" in models:
                t_low = models["xgb"].get("risk_threshold_low", 0.4)
                t_high = models["xgb"].get("risk_threshold_high", 0.7)
            else:
                t_low, t_high = 0.4, 0.7

            st.subheader("Risk Assessment")
            risk_cols = st.columns(len(scores))
            for i, (name, score) in enumerate(scores.items()):
                with risk_cols[i]:
                    c = "🔴" if score >= t_high else "🟡" if score >= t_low else "🟢"
                    lv = "HIGH" if score >= t_high else "MODERATE" if score >= t_low else "LOW"
                    label = f"{name} {'(Primary)' if name == primary else '(Baseline)'}"
                    st.metric(label, f"{c} {score:.1%}", lv)

            # ── Vital Sign Flags ────────────────────────────────────
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
                st.dataframe(pd.DataFrame(flags, columns=["Vital Sign", "Value", "Flag"]),
                             use_container_width=True, hide_index=True)
            else:
                st.success("All vital signs within normal ranges.")

            # ── RAG Recommendation ──────────────────────────────────
            st.subheader("Clinical Recommendation (RAG)")

            if api_key:
                with st.spinner("Generating recommendation..."):
                    try:
                        # Step 1: Retrieve guidelines (if RAG available)
                        context = ""
                        if rag_collection and embed_model:
                            abnormals = []
                            if hr > 90: abnormals.append("tachycardia")
                            if temp > 38.3 or temp < 36: abnormals.append("abnormal temperature")
                            if map_val < 65: abnormals.append("hypotension")
                            if resp > 22: abnormals.append("tachypnea")
                            if o2sat < 92: abnormals.append("hypoxemia")
                            if lactate > 2: abnormals.append("elevated lactate")
                            query = f"sepsis management {' '.join(abnormals)}"
                            # Encode query ourselves to avoid Streamlit deadlock
                            query_embedding = embed_model.encode([query]).tolist()
                            results = rag_collection.query(query_embeddings=query_embedding, n_results=3)
                            context = "\n---\n".join(results["documents"][0])
                            st.caption(f"Retrieved {len(context)} chars from SSC 2021 guidelines")

                        # Step 2: Build prompt
                        risk_level = "HIGH" if primary_risk >= 0.7 else "MODERATE" if primary_risk >= 0.4 else "LOW"
                        prompt = f"""You are a clinical decision support assistant. Based on the patient data and guidelines, provide 3-5 brief recommendations.

Patient: {risk_level} risk ({primary_risk:.0%}), HR {hr}, SpO2 {o2sat}%, Temp {temp}°C, MAP {map_val}, RR {resp}, Age {age}, Lactate {lactate}, WBC {wbc}, Creatinine {creatinine}

Guidelines:
{context if context else "Use standard Surviving Sepsis Campaign 2021 recommendations."}

Provide concise bullet-point recommendations. Note this is AI-generated and requires physician review."""

                        # Step 3: Call Gemini
                        from google import genai
                        from google.genai import types
                        gemini_client = genai.Client(api_key=api_key)
                        response = gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                                max_output_tokens=400,
                            ),
                        )
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error: {type(e).__name__}: {e}")

            elif rag_collection and embed_model:
                st.info("Add a Gemini API key to enable AI recommendations. Showing retrieved guidelines:")
                qe = embed_model.encode(["sepsis management"]).tolist()
                results = rag_collection.query(query_embeddings=qe, n_results=3)
                for doc in results["documents"][0]:
                    st.markdown(f"> {doc}")
            else:
                st.info("Build RAG database first: `python rag/build_vectordb.py`")

            st.divider()
            st.caption("⚠️ AI-generated clinical decision support. All recommendations require physician review.")

with tab2:
    st.header("Batch Patient Analysis")
    st.write("Upload a CSV with patient vitals.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head(), use_container_width=True)
        if st.button("Run Batch Assessment") and "xgb" in models:
            xgb_data = models["xgb"]
            X = batch_df.reindex(columns=xgb_data["feature_cols"], fill_value=0).values
            batch_df["risk_score"] = xgb_data["model"].predict_proba(X)[:, 1]
            batch_df["risk_level"] = pd.cut(batch_df["risk_score"], bins=[0, 0.4, 0.7, 1.0],
                                            labels=["LOW", "MODERATE", "HIGH"])
            st.dataframe(batch_df, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(batch_df))
            c2.metric("High Risk", (batch_df["risk_level"] == "HIGH").sum())
            c3.metric("Moderate Risk", (batch_df["risk_level"] == "MODERATE").sum())

with tab3:
    st.header("About SepsisGuard")
    st.markdown("""
    **SepsisGuard** is an AI-powered clinical decision support system for early sepsis detection.

    ### Models (trained on 40,336 ICU patients)
    | Model | Features | AUROC | Role |
    |-------|----------|-------|------|
    | **LSTM** | 21 (24hr sequences) | **0.83** | Primary |
    | **XGBoost** | 21 (snapshots) | **0.70** | Secondary |
    | **Logistic Regression** | 6 (vitals only) | **0.59** | Baseline |

    ### RAG Pipeline
    Retrieves relevant Surviving Sepsis Campaign 2021 guidelines and generates
    evidence-based recommendations using Google Gemini.

    ### Disclaimer
    For educational and research purposes only. NOT a substitute for professional medical judgment.
    """)
