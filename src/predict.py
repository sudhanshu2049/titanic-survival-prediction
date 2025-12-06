"""Simple prediction utility for the trained Titanic model.

Usage (from repo root):
  python src/predict.py --model models/model.joblib --input data/titanic.csv

If `--input` is omitted, the script predicts on the first row of `data/titanic.csv`.
"""
import argparse
import os
import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure src is importable when run from repo root
sys.path.append(os.path.dirname(__file__))


def load_model(path: str):
	return joblib.load(path)


def main(model_path: str = None, input_path: str = None):
	repo_root = Path(__file__).resolve().parents[1]
	if model_path is None:
		model_path = repo_root / 'models' / 'model.joblib'
	else:
		model_path = Path(model_path)

	if input_path is None:
		input_path = repo_root / 'data' / 'titanic.csv'
	else:
		input_path = Path(input_path)

	if not model_path.exists():
		print('Model not found at', model_path)
		return
	if not input_path.exists():
		print('Input CSV not found at', input_path)
		return

	model = load_model(str(model_path))
	df = pd.read_csv(str(input_path))

	keep_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	X = df[keep_cols]

	preds = model.predict(X)
	probs = None
	if hasattr(model, 'predict_proba'):
		probs = model.predict_proba(X)

	print('Predictions:')
	for i, p in enumerate(preds):
		line = f'Row {i}: Pred={int(p)}'
		if probs is not None:
			line += f', Prob={probs[i].round(3).tolist()}'
		print(line)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Path to model.joblib', default=None)
	parser.add_argument('--input', help='CSV input file', default=None)
	args = parser.parse_args()
	main(args.model, args.input)

