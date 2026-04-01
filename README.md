# SepsisGuard: Intelligent Clinical Decision Support System

## Project Overview
AI-powered early sepsis detection combining XGBoost prediction with SHAP explainability and RAG-based clinical recommendations for ICU settings.

## Objective
Predict sepsis onset in ICU patients with AUROC > 0.75, providing explainable risk scores and evidence-based treatment recommendations through retrieval-augmented generation.

## Dataset
- **Source**: PhysioNet Computing in Cardiology Challenge 2019
- **Size**: 40,336 ICU patients (1,552,210 hourly records)
- **Features**: 30 (16 base + 14 temporal)
- **Sepsis prevalence**: 2,932 patients (7.3%)
- **Target**: SepsisLabel (hourly sepsis indicator)

## Results

| Model | Features | AUROC | PR-AUC | Sensitivity | Specificity | F1 |
|-------|----------|-------|--------|-------------|-------------|-----|
| **XGBoost** | 30 | **0.81** | **0.10** | **71%** | **77%** | **0.09** |
| Logistic Regression | 30 | 0.72 | 0.07 | 55% | 80% | 0.08 |
| Target | - | >0.75 | - | - | - | - |

### Key Findings
- XGBoost outperforms LR by 9 AUROC points using the same 30 features
- Temporal features (vital sign deltas and rolling means) significantly improve prediction
- Dropping ultra-sparse lab features (>80% missing) reduces noise and improves performance
- Top features: ICULOS, Temp, Lactate, HospAdmTime, Resp rolling mean
- Clinical scenario evaluation: 5/6 scenarios correctly classified

## Technical Architecture

### Data Pipeline
1. **Input**: PhysioNet .psv files (40,336 patients across training_setA + training_setB)
2. **Preprocessing**: Forward-fill vitals, median-impute labs
3. **Feature Engineering**: 14 temporal features (deltas + 3-hour rolling means for 7 vitals)
4. **Output**: 30-feature matrix per hourly record

### ML Models
- **Primary — XGBoost**: 30 features, Optuna-tuned (50 trials, 3-fold CV), scale_pos_weight for class imbalance, early stopping
- **Baseline — Logistic Regression**: Same 30 features, class_weight=balanced
- **Explainability — SHAP**: Per-patient feature contribution analysis showing which vitals drive each prediction

### RAG Pipeline
1. **Knowledge Base**: 47 chunks from 3 sources:
   - Surviving Sepsis Campaign 2021 (Evans et al., Critical Care Medicine)
   - 2023 Emergency Department Sepsis Update (Guarino et al., J Clinical Medicine)
   - AAFP Practice Guidelines 2022
2. **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
3. **Vector Store**: ChromaDB (persistent, local)
4. **Generation**: Google Gemini 2.5 Flash with constrained prompting
5. **Output**: Patient-specific, evidence-based clinical recommendations

### Dashboard (Streamlit)
- Real-time vital sign input (12 clinical fields)
- XGBoost risk scoring with calibrated thresholds (LOW < 15% < MODERATE < 35% < HIGH)
- SHAP explainability — top 10 feature contributions per prediction
- RAG-powered clinical recommendations grounded in published guidelines
- Vital sign flagging (tachycardia, hypotension, hypoxemia, etc.)

## Setup

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Preprocessing
```bash
python data/preprocess.py 40336
```

### Train Models
```bash
python models/train_xgboost.py     # XGBoost (primary)
python models/baseline_lr.py       # Logistic Regression (baseline)
```

### Build RAG Database
```bash
python rag/build_vectordb.py
```

### Run Evaluation
```bash
python models/compare_models.py          # Model comparison plots
python models/clinical_evaluation.py     # Clinical scenario testing
```

### Launch Dashboard
```bash
streamlit run app.py
```
Set `GOOGLE_API_KEY` in `.env` or enter in the sidebar for AI recommendations.
Free key at https://aistudio.google.com/apikey

## Project Structure
```
sepsisguard-clinical-decision-support/
├── README.md
├── requirements.txt
├── .env                              # GOOGLE_API_KEY (not committed)
├── app.py                            # Streamlit dashboard
├── data/
│   ├── explore_data.py               # Data exploration
│   ├── preprocess.py                 # Preprocessing (supports 40K+ patients)
│   └── processed/                    # Preprocessed CSVs (gitignored)
├── models/
│   ├── train_xgboost.py              # XGBoost training + Optuna
│   ├── baseline_lr.py                # LR baseline training
│   ├── compare_models.py             # ROC, PR, confusion matrix, threshold analysis
│   ├── clinical_evaluation.py        # Clinical scenario testing
│   ├── xgboost_model.pkl             # Trained XGBoost (gitignored)
│   └── baseline_lr.pkl               # Trained LR (gitignored)
├── rag/
│   ├── sepsis_guidelines.txt         # SSC 2021 + 2023 ED Update + AAFP 2022
│   ├── build_vectordb.py             # ChromaDB builder
│   ├── rag_engine.py                 # RAG retrieval + Gemini generation
│   └── chroma_db/                    # Vector store (gitignored)
└── results/
    ├── xgboost_results.png           # ROC, PR, features, score distribution
    ├── model_comparison_roc.png      # XGBoost vs LR comparison
    └── baseline_lr_roc.png           # LR ROC curve
```

## References
1. Reyna, M., et al. (2019). Early Prediction of Sepsis from Clinical Data. Critical Care Medicine, 48(2), 210-217.
2. Evans, L., et al. (2021). Surviving Sepsis Campaign: International Guidelines. Intensive Care Medicine, 47(11), 1181-1247.
3. Guarino, M., et al. (2023). 2023 Update on Sepsis Management in the Emergency Department. J Clinical Medicine, 12(9), 3188.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
5. Lundberg, S., & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
