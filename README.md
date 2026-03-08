# Speech Speed and Tempo Classification

Speech Speed and Tempo Classification is a **machine learning project** that detects **speed** and **tempo** alterations in speech recordings using **FBANK features** and classical classifiers implemented in **Python** with **scikit-learn**.

The project evaluates how well different models classify modified speech recordings and highlights the importance of feature design when temporal information is removed through mean pooling.

---

## Key Features

✔ Speech speed alteration classification  
✔ Speech tempo alteration classification  
✔ Classical machine learning benchmarking  
✔ Feature preprocessing with temporal mean pooling  
✔ Hyperparameter tuning for k-Nearest Neighbours  
✔ Comparative evaluation of multiple classifiers  
✔ Error analysis of feature limitations  
✔ Reproducible Python-based ML workflow

---

## Technologies Used

- **Python**
- **scikit-learn**
- **NumPy**
- **Joblib**
- **uv** for environment management
- **Object-oriented / modular experiment structure**

---

## Project Architecture

The project is organized into separate components for **data handling, model training, evaluation, and saved baseline models**.

### Feature & Data Pipeline

- Pre-computed **FBANK** features
- Temporal mean pooling to convert variable-length inputs into fixed-length vectors
- Separate data and model directories for reproducibility

### Models Evaluated

- `k-Nearest Neighbours (kNN)`
- `Logistic Regression`
- `Linear SVM`
- `Random Forest`

### Evaluation Focus

- Speed classification
- Tempo classification
- Comparison against provided baselines
- Analysis of why tempo remains difficult under temporally pooled features

---

## Key Results

- Improved **speed classification accuracy** from **79.2%** to **86.6%**
- Benchmarked multiple classifiers for **tempo classification**
- Best tempo accuracy reached **31.4%**
- Identified a key limitation: **temporal mean pooling removes timing structure needed for tempo discrimination**

---

## Results Summary

### Speed Classification

| Configuration       | Accuracy (%) |
| ------------------- | -----------: |
| Baseline (kNN, k=1) |         79.2 |
| + StandardScaler    |         81.0 |
| kNN (k=3)           |         82.4 |
| kNN (k=5)           |         84.0 |
| kNN (k=7)           |     **86.6** |

### Tempo Classification

| Model / Configuration | Accuracy (%) |
| --------------------- | -----------: |
| Baseline (kNN, k=1)   |         24.6 |
| kNN + StandardScaler  |         24.2 |
| kNN (k=3)             |         24.6 |
| kNN (k=5)             |         24.8 |
| kNN (k=7)             |         24.8 |
| kNN (k=5, no scaling) |         26.8 |
| Logistic Regression   |         27.8 |
| Linear SVM            |         30.0 |
| Random Forest         |     **31.4** |

---

## Methods

### Feature Representation

- Input features are pre-computed **FBANK** representations
- Each sample is reshaped and temporally pooled into a fixed-length **64-dimensional vector**
- Temporal mean pooling provides a lightweight representation but removes frame-level timing information

### Improvements Explored

- Feature standardization using `StandardScaler`
- Hyperparameter tuning for `k` in kNN
- Comparison of multiple classifiers under the same pooled-feature setup
- Model-size control to satisfy evaluation constraints

---

## Discussion

The experiments show that classical machine learning models can effectively detect **speed modifications** when combined with feature standardization and simple hyperparameter tuning.

In contrast, **tempo classification** remains significantly more difficult because temporal mean pooling removes the timing information required to distinguish tempo changes. This project demonstrates how **feature engineering and representation choices strongly affect downstream model performance**.

---

## Project Structure

```text
speech-speed-tempo-classification
│
├── data/                      # Dataset files (not included by default)
├── models/
│   ├── baseline_fbank/
│   └── baseline_signal/
├── src/
│   ├── baseline_fbank/
│   ├── baseline_signal/
│   ├── demo.ipynb
│   └── evaluate.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Running the Project

### Requirements

- Python
- `uv` environment manager

### Setup

Install dependencies:

```bash
uv sync
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Run evaluation or experiments using the relevant scripts in `src/`.

---

## Data

The original coursework used dataset files stored separately from the repository.  
Place the required data files inside the `data/` directory before running experiments.

---

## Future Improvements

- Preserve temporal structure instead of averaging across frames
- Add additional statistics such as variance alongside mean pooling
- Use full `64 × T` FBANK representations
- Explore sequence-aware or deep learning approaches for tempo-sensitive classification
- Add result visualizations directly into the repository README

---

## Skills Demonstrated

- Python
- scikit-learn
- Feature preprocessing
- Hyperparameter tuning
- Model benchmarking
- Error analysis
- Experimental comparison
- Reproducible ML workflows

---

## Notes

This project was originally developed as part of coursework in the **MSc Artificial Intelligence** programme at the **University of Sheffield**, and has been refined and presented here as a portfolio project focused on **applied machine learning experimentation and analysis**.

---

## Author

**Cheng-Yuan King**  
MSc Artificial Intelligence – University of Sheffield
