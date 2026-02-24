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

### ✅ Completed (Phase 1)
- [x] Data acquisition and exploration
- [x] Preprocessing pipeline with missing value handling
- [x] Early warning label creation (6-hour prediction window)
- [x] Baseline model training (Logistic Regression)
- [x] LSTM architecture definition

### 📊 Current Results

| Model | Features | AUROC | Parameters | Status |
|-------|----------|-------|------------|--------|
| **Logistic Regression** | Current vitals (6 features) | **0.7074** | - | ✅ Complete |
| **Random Forest** | Current vitals (6 features) | 0.5955 | - | ⚠️ Overfitting issues |
| **LSTM** | 24-hour sequences | TBD | 53,825 | 🔄 Architecture ready |
| **Target** | - | **> 0.75** | - | 🎯 Goal |

### 🔍 Key Findings
- **Sepsis prevalence**: 9% of patients (90/1000) - realistic clinical distribution
- **Class imbalance**: 1.16% positive samples (450/38,809 hours)
- **Baseline performance**: LR achieves 0.71 AUROC with 69% recall
- **Missing data**: 21% reduction after preprocessing (forward-fill vitals, median imputation labs)

### 🔄 In Progress (Phase 2)
- [ ] LSTM training on 24-hour vital sign sequences
- [ ] Hyperparameter tuning and evaluation
- [ ] Scale to full 40,000+ patient dataset

### 📋 Planned (Phase 3-4)
- [ ] RAG pipeline with Weaviate vector database
- [ ] LLM integration (Google Gemini) for explainable recommendations
- [ ] Streamlit dashboard development
- [ ] End-to-end system evaluation

## Technical Architecture

### Data Pipeline
1. **Input**: PhysioNet .psv files (pipe-separated values)
2. **Preprocessing**: Missing value handling, feature engineering, early warning labels
3. **Output**: Clean dataset ready for ML training

### Models
- **Baseline**: Logistic Regression (current vitals only)
- **Advanced**: LSTM (24-hour temporal sequences)
  - 2-layer LSTM (64 hidden units)
  - Dropout regularization (0.3)
  - Fully connected layers (64→32→1)
  - Sigmoid output (risk probability)

## Setup

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib torch
```

### Run Preprocessing
```bash
python data/preprocess.py
```

### Train Baseline Model
```bash
python models/baseline_lr.py
```

### Test LSTM Architecture
```bash
python models/lstm_model.py
```

## Project Structure

```
Capstone sepsis guard/
├── README.md
├── data/
│   ├── explore_data.py          # Data exploration (list .psv files, sample stats)
│   ├── preprocess.py            # Preprocessing pipeline (missing values, early labels)
│   └── processed/
│       └── preprocessed_1000.csv # Preprocessed training data (1000 patients)
├── models/
│   ├── baseline_lr.py           # Logistic Regression baseline training
│   ├── baseline_lr.pkl          # Trained LR model (joblib)
│   └── lstm_model.py            # LSTM architecture (PyTorch)
├── results/
│   └── baseline_lr_roc.png      # ROC curve for baseline model
└── venv/                        # Python virtual environment
```

**Note:** PhysioNet .psv data lives under an external path (`training_setA` / `training_setB`); configure `DATA_ROOT` or the path in `preprocess.py` / `explore_data.py` to point to your data directory.
