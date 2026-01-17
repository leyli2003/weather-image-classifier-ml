# Weather Condition Recognition from Images (Classical Machine Learning)

This repository documents a course project on **weather condition recognition from real-world images** using **classical machine learning** (no CNNs / deep learning). The focus is on building and evaluating **supervised scikit-learn pipelines** under **non-controlled imaging conditions** such as illumination changes, viewpoint variability, background clutter, and intra-class diversity.

## Motivation
Visual recognition “in the wild” is challenging because the same weather condition can look very different depending on lighting, camera angle, scene context, and image quality. This project explores how far **feature-based classical ML methods** can go on this task, and what limitations appear without learned deep visual representations.

## Dataset
The full dataset used in this project was **provided as part of a university course** and is **not redistributed** in this repository.

To still communicate the project clearly, this repo includes:
- a small `sample_images/` subset for demonstration
- label mapping in `labelClasses.csv`

If you have access to the original dataset, place it locally in a folder named `images/` (or adjust paths in the scripts accordingly).

**Classes (label → name)**:
- `0` → `cloudy`
- `1` → `rain`
- `2` → `shine`
- `3` → `sunrise`

## Methodology
An end-to-end supervised workflow was designed and evaluated:

- Data preparation and label handling
- Classical feature extraction from images
- Model training using standardized **scikit-learn pipelines**
- Hyperparameter tuning via **GridSearchCV**
- Comparative evaluation across multiple classifiers
- Optional model serialization with `joblib` for reproducibility

The emphasis is on **robustness and generalization** rather than dataset-specific optimization.

## Models
The study evaluates multiple classical classifiers implemented with consistent pipelines:

- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **Multilayer Perceptron (MLP)**

Grid search scripts are provided to tune key hyperparameters for each model.

## Repository Structure
- `cleanup.py` — preprocessing / dataset preparation utilities  
- `svm.py`, `randomForest.py`, `mlp.py` — training scripts  
- `svmGridSearch.py`, `randomForestGridSearch.py`, `mlpGridSearch.py` — hyperparameter tuning (GridSearchCV)  
- `labelClasses.csv` — class label mapping  
- `sample_images/` — small demo subset of the dataset (for illustration only)  
- `assets/` — figures/results (e.g., confusion matrices, tables)

## How to Run (example)
> Exact commands may vary depending on how your paths are set in the scripts.

1. (Optional) Run preprocessing:
```bash
python cleanup.py
