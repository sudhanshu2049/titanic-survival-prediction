"""Model pipeline factory for Titanic classifier.

Provides a sklearn `Pipeline` that handles simple preprocessing
and a RandomForest classifier.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(random_state: int = 42) -> Pipeline:
	numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
	categorical_features = ['Pclass', 'Sex', 'Embarked']

	numeric_transformer = Pipeline([
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler()),
	])

	categorical_transformer = Pipeline([
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore')),
	])

	preprocessor = ColumnTransformer([
		('num', numeric_transformer, numeric_features),
		('cat', categorical_transformer, categorical_features),
	])

	clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

	pipeline = Pipeline([
		('preprocessor', preprocessor),
		('classifier', clf),
	])

	return pipeline


if __name__ == '__main__':
	p = build_pipeline()
	print('Pipeline built:', p)

