#!/usr/bin/env python3
"""SepsisGuard - Clinical Decision Support Dashboard"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from dotenv import load_dotenv

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
    xgb_path = PROJECT_ROOT / "models" / "xgboost_model.pkl"
    lr_path = PROJECT_ROOT / "models" / "baseline_lr.pkl"
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)
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
tab1, tab2 = st.tabs(["Patient Assessment", "About"])

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

        # ── Build feature vector (base + temporal) ──────────────────
        # For single-point input: deltas=0, rolling=current value
        for col in ["HR", "O2Sat", "Temp", "MAP", "Resp", "SBP", "DBP"]:
            patient[f"{col}_delta"] = 0.0
            patient[f"{col}_roll3"] = patient.get(col, 0)

        # ── Risk Scores ─────────────────────────────────────────────
        scores = {}

        if "xgb" in models:
            xgb_data = models["xgb"]
            X = np.array([[patient.get(c, 0) for c in xgb_data["feature_cols"]]])
            scores["XGBoost"] = float(xgb_data["model"].predict_proba(X)[0][1])

        # LR kept for evaluation only, not shown on dashboard

        if not scores:
            st.error("No models available.")
        else:
            primary = list(scores.keys())[0]
            primary_risk = scores[primary]

            # Get risk thresholds from XGBoost model (or use defaults)
            # Clinically meaningful thresholds (validated via scenario evaluation)
            t_low = 0.15   # below 15% = LOW
            t_high = 0.35  # above 35% = HIGH

            st.subheader("Risk Assessment")
            risk_cols = st.columns(len(scores))
            for i, (name, score) in enumerate(scores.items()):
                with risk_cols[i]:
                    c = "🔴" if score >= t_high else "🟡" if score >= t_low else "🟢"
                    lv = "HIGH" if score >= t_high else "MODERATE" if score >= t_low else "LOW"
                    label = f"{name} {'(Primary)' if name == primary else '(Baseline)'}"
                    st.metric(label, f"{c} {score:.1%}", lv)

            # ── SHAP Explanation ────────────────────────────────────
            if "xgb" in models:
                st.subheader("Why This Prediction? (SHAP)")
                try:
                    import shap
                    xgb_model = models["xgb"]["model"]
                    feature_names = models["xgb"]["feature_cols"]
                    X_df = pd.DataFrame([patient.get(c, 0) for c in feature_names],
                                        index=feature_names, columns=["value"]).T
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_values = explainer.shap_values(X_df)

                    # Get top contributing features — exclude temporal (delta/roll3) since
                    # they're meaningless on single-point dashboard input
                    sv = shap_values[0]
                    base_indices = [i for i, name in enumerate(feature_names)
                                    if "_delta" not in name and "_roll3" not in name]
                    base_sv = [(i, sv[i]) for i in base_indices]
                    base_sv.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_features = base_sv[:10]

                    # Build display
                    shap_data = []
                    for idx, val in top_features:
                        name = feature_names[idx]
                        direction = "Increases risk" if val > 0 else "Decreases risk"
                        shap_data.append({
                            "Feature": name,
                            "Value": f"{X_df.iloc[0, idx]:.1f}",
                            "Impact": f"{'+' if val > 0 else ''}{val:.3f}",
                            "Direction": direction,
                        })

                    shap_df = pd.DataFrame(shap_data)
                    st.dataframe(shap_df, use_container_width=True, hide_index=True)

                    # Bar chart
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    top_names = [feature_names[i] for i, _ in top_features][::-1]
                    top_vals = [v for _, v in top_features][::-1]
                    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals]
                    ax.barh(top_names, top_vals, color=colors)
                    ax.set_xlabel("SHAP Value (impact on prediction)")
                    ax.set_title("Top 10 Feature Contributions")
                    ax.axvline(0, color="black", linewidth=0.5)
                    ax.grid(alpha=0.3, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")

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
                            results = rag_collection.query(query_embeddings=query_embedding, n_results=5)
                            context = "\n---\n".join(results["documents"][0])
                            st.caption(f"Retrieved {len(context)} chars from clinical guidelines (SSC 2021, 2023 ED Update, AAFP 2022)")

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
    st.header("About SepsisGuard")
    st.markdown("""
    **SepsisGuard** is an AI-powered clinical decision support system for early sepsis detection.

    ### Models (trained on 40,336 ICU patients, 30 features each)
    | Model | Features | AUROC | Role |
    |-------|----------|-------|------|
    | **XGBoost** | 30 (vitals + labs + demographics + temporal) | **0.81** | Primary |
    | **Logistic Regression** | 30 (same features) | **0.72** | Baseline |

    ### RAG Pipeline
    Retrieves relevant Surviving Sepsis Campaign 2021 guidelines and generates
    evidence-based recommendations using Google Gemini.

    """)
