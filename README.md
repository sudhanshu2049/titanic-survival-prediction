# Titanic Survival Prediction

This repository contains a small, self-contained example for training a
Titanic survival classifier and running predictions.

Contents:
- `data/titanic.csv` - small sample dataset (replace with the full dataset if desired).
- `src/` - scripts for preprocessing, training, and predicting.
- `models/` - trained model artifacts (created by `src/train.py`).
- `results/` - textual training report and metrics.

Quick start
1. Create a virtual environment and install dependencies:

	```powershell
	python -m venv .venv; .\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

2. Train the model (reads `data/titanic.csv`):

	```powershell
	python src/train.py
	```

	This writes `models/model.joblib` and `results/report.txt`.

3. Predict (example):

	```powershell
	python src/predict.py --model models/model.joblib --input data/titanic.csv
	```

Notes
- The `src/` scripts are lightweight examples intended for education and
  quick experimentation. They use a RandomForest classifier with a small
  preprocessing pipeline. Replace or extend the code for production use.
