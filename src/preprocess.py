"""Data loading helpers for the Titanic project.

Functions:
 - load_data(path): loads CSV and returns (X, y)
"""
from typing import Tuple, Optional
import pandas as pd


def load_data(path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
	"""Load dataset from CSV and return features (X) and label (y).

	If the CSV contains a `Survived` column, it will be returned as `y`.
	"""
	df = pd.read_csv(path)
	if 'Survived' in df.columns:
		y = df['Survived']
		X = df.drop(columns=['Survived'])
	else:
		y = None
		X = df
	return X, y


if __name__ == '__main__':
	import os
	here = os.path.dirname(__file__)
	sample = os.path.normpath(os.path.join(here, '..', 'data', 'titanic.csv'))
	X, y = load_data(sample)
	print(f"Loaded X shape={X.shape}, y present={y is not None}")
