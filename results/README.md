Results directory
-----------------

This folder contains outputs produced by `src/train.py` and other scripts:

- `report.txt` - textual training report with metrics (accuracy, classification report).
- `models/model.joblib` - trained model artifact (created after training).

Note: `results/report.txt` is initialized with an example and will be overwritten
whenever you run `python src/train.py`.
