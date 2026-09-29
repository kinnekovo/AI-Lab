# Experiment 1 - Text Classification

## 1. Task

10-class news text classification using TF-IDF and classical ML / MLP models.

## 2. Dataset

- 7,368 labeled training samples
- 2,457 unlabeled test samples
- Stratified 8:2 validation split: 5,894 training samples and 1,474 validation samples
- `random_state=42`

## 3. Project Structure

- `data/`: immutable labeled training data and unlabeled test data.
- `experiments/`: numbered, independently executable experiment scripts.
- `results/`: CSV records from tuning and comparison experiments; error-analysis CSV files are under `results/error_analysis/`.
- `outputs/`: final submission artifacts, including predictions, final configuration, and the confusion-matrix figure.
- `docs/`: experiment notes and report materials.
- `final_train_predict.py`: trains the validated final MLP on all labeled data and creates submission predictions.

## 4. Experimental Pipeline

Baseline → model tuning → TF-IDF optimization → final fair comparison → error analysis → full-data final training.

## 5. Final Configuration

TF-IDF:

- `max_features=20000`
- `ngram_range=(1, 1)`

MLP:

- `hidden_layer_sizes=(100,)`
- `activation='relu'`
- `alpha=0.001`
- `max_iter=300`
- `random_state=42`

Validation results:

- Accuracy = 0.9335
- Macro-F1 = 0.9340

## 6. Run Final Prediction

From the project root:

```bash
python final_train_predict.py
```

The script writes `outputs/predictions.csv` and `outputs/final_training_config.csv`.

## 7. Reproduce Individual Experiments

From the project root:

```bash
python experiments/01_baseline.py
python experiments/09_final_comparison.py
python experiments/11_error_analysis.py
```

Some MLP experiments take a comparatively long time to run. Each experiment resolves paths from its own file location, so it does not depend on the terminal's current working directory.

## 8. Main Results

| Model | Macro-F1 |
| --- | ---: |
| MLP | 0.9340 |
| Logistic Regression | 0.9311 |
| RBF SVM | 0.9268 |
| Linear SVM | 0.9239 |
