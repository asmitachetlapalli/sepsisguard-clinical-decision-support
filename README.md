# SepsisGuard: Intelligent Clinical Decision Support System

## Project Overview
AI-powered early sepsis detection combining LSTM time-series prediction with RAG-based explainable recommendations for ICU settings.


## Objective
Predict sepsis onset 4-6 hours before clinical diagnosis with AUROC > 0.75, providing evidence-based treatment recommendations through retrieval-augmented generation.

## Dataset
- **Source**: PhysioNet Challenge 2019
- **Size**: 40,336 ICU patients (using 1,000 for initial development)
- **Features**: 8 vital signs + 26 lab values + 6 demographics
- **Target**: Early sepsis detection (6-hour lead time)

## Progress Update

### Phase 1 - Data & Baseline
- Data acquisition and exploration
- Preprocessing pipeline with missing value handling
- Early warning label creation (6-hour prediction window)
- Baseline model training (Logistic Regression)

### Phase 2 - LSTM Training
- LSTM architecture with temporal feature engineering (deltas, rolling means)
- Training pipeline with weighted sampling for class imbalance
- Early stopping and cosine annealing learning rate schedule
- 8-hour sliding window sequences with 22 engineered features

### Phase 3 - RAG Pipeline
- ChromaDB vector database with Surviving Sepsis Campaign 2021 guidelines
- Sentence-transformer embeddings (all-MiniLM-L6-v2) for semantic search
- Context retrieval engine for patient-specific guideline matching

### Phase 4 - Integration & Dashboard
- Google Gemini (free tier) integration for explainable AI recommendations
- Streamlit dashboard with real-time patient assessment
- Batch analysis for multiple patients via CSV upload
- Vital sign flagging with clinical thresholds

### Current Results

| Model | Features | AUROC | Parameters | Status |
|-------|----------|-------|------------|--------|
| **Logistic Regression** | Current vitals (6 features) | **0.7074** | - | Complete |
| **LSTM** | 8-hour sequences (22 temporal features) | **0.6574** | 8,257 | Complete |
| **Random Forest** | Current vitals (6 features) | 0.5955 | - | Overfitting |
| **Target** | - | **> 0.75** | - | Goal |

### Key Findings
- **Sepsis prevalence**: 9% of patients (90/1000) - realistic clinical distribution
- **Class imbalance**: 1.16% positive samples (450/38,809 hours)
- **Baseline performance**: LR achieves 0.71 AUROC with 69% recall - outperforms LSTM on small dataset
- **LSTM insight**: Deep learning underperforms LR with only 90 sepsis patients; scaling to full 40K dataset expected to improve results significantly
- **Missing data**: 21% reduction after preprocessing (forward-fill vitals, median imputation labs)

## Technical Architecture

### Data Pipeline
1. **Input**: PhysioNet .psv files (pipe-separated values)
2. **Preprocessing**: Missing value handling, feature engineering, early warning labels
3. **Output**: Clean dataset ready for ML training

### Models
- **Baseline**: Logistic Regression (current vitals only)
- **Advanced**: LSTM with temporal features
  - 1-layer LSTM (32 hidden units)
  - Temporal features: vital sign deltas + 3-hour rolling means
  - Weighted random sampling for class imbalance
  - Dropout regularization (0.5)
  - Sigmoid output (risk probability)

### RAG Pipeline
1. **Knowledge Base**: Surviving Sepsis Campaign 2021 guidelines (22 chunks)
2. **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers
3. **Vector Store**: ChromaDB (persistent, local)
4. **Generation**: Google Gemini 2.0 Flash (free tier)
5. **Output**: Evidence-based, patient-specific clinical recommendations

## Setup

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Preprocessing
```bash
python data/preprocess.py
```

### Train Models
```bash
python models/baseline_lr.py    # Logistic Regression
python models/train_lstm.py     # LSTM
```

### Build RAG Database
```bash
python rag/build_vectordb.py
```

### Launch Dashboard
```bash
streamlit run app.py
```
Set `GOOGLE_API_KEY` env var or enter it in the sidebar for AI-generated recommendations.
Get a free key at https://aistudio.google.com/apikey

## Project Structure

```
sepsisguard-clinical-decision-support/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit dashboard
├── data/
│   ├── explore_data.py             # Data exploration
│   ├── preprocess.py               # Preprocessing pipeline
│   └── processed/
│       └── preprocessed_1000.csv   # Preprocessed training data
├── models/
│   ├── baseline_lr.py              # LR baseline training
│   ├── baseline_lr.pkl             # Trained LR model
│   ├── lstm_model.py               # LSTM architecture (PyTorch)
│   ├── train_lstm.py               # LSTM training pipeline
│   └── lstm_trained.pth            # Trained LSTM model
├── rag/
│   ├── sepsis_guidelines.txt       # Surviving Sepsis Campaign guidelines
│   ├── build_vectordb.py           # ChromaDB builder
│   ├── rag_engine.py               # RAG retrieval + generation
│   └── chroma_db/                  # Persistent vector store
└── results/
    ├── baseline_lr_roc.png         # LR ROC curve
    └── lstm_roc.png                # LSTM ROC curve + score distribution
```

**Note:** PhysioNet .psv data lives under an external path (`training_setA` / `training_setB`); configure `DATA_ROOT` or the path in `preprocess.py` / `explore_data.py` to point to your data directory.
