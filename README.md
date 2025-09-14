# spam-ham

This project applies **4 machine learning classifiers** to the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) to detect spam vs. non-spam emails.  
It was developed in a **Jupyter notebook** using Python and scikit-learn, and includes full evaluation, comparisons, visualizations, and a short PDF presentation (inside spam-ham folder).

---

## Overview

- **Dataset: Spambase (UCI / OpenML)**
  - 57 numerical features describing frequency of words, characters, and patterns in emails.
  - Target column: `class` → 1 (spam), 0 (non-spam).

- **Models trained**
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  

- **Evaluation metrics**
  - Accuracy (train/test)
  - Precision, Recall, F1-score
  - ROC AUC

- **Visualizations**
  - Confusion matrices
  - Individual ROC curves
  - Combined ROC curve
  - F1-score comparison
  - AUC comparison
  - Overfitting check (train vs test accuracy)

**Result:** Gradient Boosting and Random Forest achieved the highest performance, with Gradient Boosting slightly ahead in F1 and AUC.

---

## How It Works

```
1) Load dataset (Spambase via OpenML)
 ├─ Drop missing values (none found)
 ├─ Split into train (70%) and test (30%) [stratified]
 └─ Standardize features with StandardScaler

2) Define classifiers
 ├─ Logistic Regression (max_iter=200)
 ├─ Decision Tree (max_depth=10)
 ├─ Random Forest (n_estimators=100, max_depth=10)
 └─ Gradient Boosting (n_estimators=150, lr=0.1)

3) Train & evaluate
 ├─ Compute Accuracy, Precision, Recall, F1, AUC
 ├─ Save predictions & probabilities
 └─ Build summary performance table

4) Generate plots
 ├─ Confusion matrices
 ├─ ROC curves (individual + combined)
 ├─ Barplots: F1, AUC
 └─ Accuracy (train vs test)

5) Final test
 └─ Predict on a random email from test set with all 4 models
```

---

## Repo Layout

```
spam-ham/
├── plots/                         # Generated figures
│   ├── regresie_logistica_*.png
│   ├── arbore_de_decizie_*.png
│   ├── random_forest_*.png
│   ├── gradient_boosting_*.png
│   ├── comparatie_f1_score.png
│   ├── comparatie_auc.png
│   ├── roc_all_models.png
│   └── acuratete_antrenare_vs_testare.png
├── spambase/                      # Dataset files
│   ├── spambase.data
│   ├── spambase.names
│   └── spambase.DOCUMENTATION
├── notebook.ipynb                 # Main Jupyter notebook
├── Prezentare.pdf                 # Short presentation
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tabulate`

### Install deps
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```

---

## Usage

Run the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

It will:
- Train and evaluate all 4 classifiers
- Print a performance table
- Save all plots in `plots/`
- Perform a random test on one email from the test set

---

## Results

### Performance table (excerpt)
| Model              | Train Acc | Test Acc | Precision | Recall | F1  | AUC  |
|--------------------|-----------|----------|-----------|--------|-----|------|
| Gradient Boosting  | 0.969     | 0.950    | 0.949     | 0.931  | 0.940 | 0.987 |
| Random Forest      | 0.967     | 0.946    | 0.961     | 0.906  | 0.933 | 0.985 |
| Logistic Regression| 0.927     | 0.923    | 0.935     | 0.877  | 0.905 | 0.974 |
| Decision Tree      | 0.969     | 0.911    | 0.917     | 0.865  | 0.890 | 0.922 |

### Example plots
- `comparatie_f1_score.png` – barplot of F1-scores
- `comparatie_auc.png` – barplot of AUC values
- `roc_all_models.png` – combined ROC curves
- `acuratete_antrenare_vs_testare.png` – overfitting check

---

## Practical Test

The notebook also tests a random email from the test set.  
Each model predicts spam/non-spam and gives probability of spam.

Example:
```
Eticheta reală: spam
- Logistic Regression: spam (65%)
- Decision Tree: spam (98%)
- Random Forest: spam (73%)
- Gradient Boosting: spam (94%)
```

---

## License

MIT
