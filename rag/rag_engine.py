#!/usr/bin/env python3
"""
RAG engine: retrieves relevant sepsis guidelines from ChromaDB and
generates explainable recommendations using Google Gemini (free tier).
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
try:
    from google import genai as genai_new
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_SDK = False


CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "sepsis_guidelines"


def get_collection():
    """Load the ChromaDB collection with the same embedding function."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve_context(query: str, n_results: int = 5) -> str:
    """Retrieve relevant guideline chunks for a query."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs)


def build_prompt(patient_data: dict, risk_score: float, context: str) -> str:
    """Build the LLM prompt with patient data + retrieved guidelines."""
    risk_level = "HIGH" if risk_score >= 0.7 else "MODERATE" if risk_score >= 0.4 else "LOW"

    return f"""You are a clinical decision support assistant for ICU sepsis management.
Based on the patient's data and relevant clinical guidelines, provide a brief, actionable recommendation.

## Patient Data
- Sepsis Risk Score: {risk_score:.1%} ({risk_level} RISK)
- Heart Rate: {patient_data.get('HR', 'N/A')} bpm
- SpO2: {patient_data.get('O2Sat', 'N/A')}%
- Temperature: {patient_data.get('Temp', 'N/A')} °C
- MAP: {patient_data.get('MAP', 'N/A')} mmHg
- Respiratory Rate: {patient_data.get('Resp', 'N/A')} breaths/min
- Age: {patient_data.get('Age', 'N/A')} years
- Lactate: {patient_data.get('Lactate', 'N/A')} mmol/L
- WBC: {patient_data.get('WBC', 'N/A')} x10^3/μL
- Creatinine: {patient_data.get('Creatinine', 'N/A')} mg/dL

## Relevant Clinical Guidelines
{context}

## Instructions
1. Assess the patient's current status based on vitals and risk score.
2. Identify which vital signs are abnormal and what they indicate.
3. Provide 3-5 specific, prioritized clinical recommendations based on the guidelines.
4. Flag any critical values that require immediate attention.

Keep the response concise and clinically relevant. Use bullet points.
Do NOT diagnose — provide decision support only. Always note this is AI-generated and requires physician review.
"""


def generate_recommendation(
    patient_data: dict,
    risk_score: float,
    api_key: str | None = None,
) -> str:
    """
    Full RAG pipeline: retrieve context → build prompt → generate recommendation.
    """
    # Build query from patient context
    abnormals = []
    if patient_data.get("HR", 0) > 90:
        abnormals.append("tachycardia")
    if patient_data.get("Temp", 37) > 38.3 or patient_data.get("Temp", 37) < 36:
        abnormals.append("abnormal temperature")
    if patient_data.get("MAP", 70) < 65:
        abnormals.append("hypotension low MAP")
    if patient_data.get("Resp", 16) > 22:
        abnormals.append("tachypnea high respiratory rate")
    if patient_data.get("O2Sat", 98) < 92:
        abnormals.append("hypoxemia low oxygen")
    if patient_data.get("Lactate", 0) > 2:
        abnormals.append("elevated lactate")

    query = f"sepsis management {' '.join(abnormals)} risk score {risk_score:.0%}"
    if risk_score >= 0.7:
        query += " hour-1 bundle vasopressor fluid resuscitation"

    # Retrieve relevant guidelines
    context = retrieve_context(query)

    # Build prompt
    prompt = build_prompt(patient_data, risk_score, context)

    # Generate with Gemini
    if USE_NEW_SDK:
        client = genai_new.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    else:
        if api_key:
            genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY env var to test LLM generation.")
        print("Testing retrieval only...\n")

        context = retrieve_context("sepsis tachycardia hypotension treatment")
        print("Retrieved context:")
        print(context[:500])
    else:
        # Test with sample patient
        sample = {
            "HR": 110, "O2Sat": 91, "Temp": 38.9,
            "MAP": 58, "Resp": 26, "Age": 72,
            "Lactate": 4.2, "WBC": 18.5, "Creatinine": 2.1,
        }
        result = generate_recommendation(sample, risk_score=0.85, api_key=api_key)
        print(result)
