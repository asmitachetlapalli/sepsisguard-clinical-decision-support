# Setup Guide

## Prerequisites
- Python 3.10+
- Google Gemini API key (free at https://aistudio.google.com/apikey)

## Installation
1. Clone the repo: `git clone https://github.com/asmitachetlapalli/sepsisguard-clinical-decision-support.git`
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Data
1. Download PhysioNet 2019 data from https://physionet.org/content/challenge-2019/
2. Place .psv files in `data/archive/training_setA/` and `data/archive/training_setB/`
3. Run preprocessing: `python data/preprocess.py 40336`

## Train Models
1. `python models/train_xgboost.py`
2. `python models/baseline_lr.py`

## Build RAG
1. `python rag/build_vectordb.py`

## Run Dashboard
1. Add API key to `.env`: `GOOGLE_API_KEY=your_key_here`
2. `streamlit run app.py`
3. Open http://localhost:8501