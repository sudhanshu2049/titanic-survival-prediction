"""Train a Titanic survival classifier and write a short report.

Run from repo root with:
	python src/train.py

The script will read `data/titanic.csv`, train a model, save it to
`models/model.joblib` and write `results/report.txt`.
"""
import os
import sys
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure src is importable when run from repository root
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data
from model import build_pipeline


def main(data_path: str = None):
	repo_root = Path(__file__).resolve().parents[1]
	if data_path is None:
		data_path = repo_root / 'data' / 'titanic.csv'
	else:
		data_path = Path(data_path)

	print('Loading data from', data_path)
	X, y = load_data(str(data_path))
	if y is None:
		print('No `Survived` column found in dataset. Cannot train.')
		return

	# keep only columns our pipeline expects (ignore Name, Ticket, Cabin, PassengerId)
	keep_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	X = X[keep_cols]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	pipeline = build_pipeline()
	print('Fitting model...')
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	acc = accuracy_score(y_test, preds)
	report = classification_report(y_test, preds)

	# Save model
	models_dir = repo_root / 'models'
	models_dir.mkdir(parents=True, exist_ok=True)
	model_path = models_dir / 'model.joblib'
	joblib.dump(pipeline, model_path)
	print('Saved model to', model_path)

	# Save report
	results_dir = repo_root / 'results'
	results_dir.mkdir(parents=True, exist_ok=True)
	report_path = results_dir / 'report.txt'
	with open(report_path, 'w', encoding='utf8') as f:
		f.write('Titanic model training report\n')
		f.write('============================\n\n')
		f.write(f'Model path: {model_path}\n')
		f.write(f'Accuracy: {acc:.4f}\n\n')
		f.write('Classification report:\n')
		f.write(report)

	print('\nTraining complete. Metrics:')
	print(f'  Accuracy: {acc:.4f}')
	print(report)
	print('Report written to', report_path)


if __name__ == '__main__':
	main()

