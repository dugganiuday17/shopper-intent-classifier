# ShopperSense — Predicting Online Purchase Intent

ML Assignment 2 · M.Tech (AIML) · Machine Learning

- **GitHub repository:** https://github.com/<your-github-username>/shopper-intent-classifier
- **Live Streamlit app:** https://<your-app-name>.streamlit.app

---

## a. Problem statement

An e-commerce site records how each visitor browses during a session: how many pages of each
type they open, how long they stay on them, Google-Analytics-style bounce / exit / page-value
metrics, the month, whether the day is close to a holiday, the visitor's OS, browser, region,
traffic source and whether they are a new or returning visitor.

**Goal:** given these session-level signals, predict whether the session will end in a
purchase (`Revenue = 1`) or not (`Revenue = 0`). This is a **binary classification** problem
with a strong class imbalance (only ~15.6% of sessions convert), so in addition to accuracy we
care about precision, recall, F1, AUC and MCC.

Five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive
Bayes and an RBF Support Vector Machine — are trained on the same data, compared on the same
hold-out test set, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, dataset id 468 — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Original reference | Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Neural Computing & Applications* 31, 6893–6908 (2019) |
| Instances | 12,330 sessions (12,205 after dropping 125 exact duplicate rows) |
| Input features | 17 (10 numeric, 7 categorical/binary) |
| Target | `Revenue` — TRUE (purchase) / FALSE (no purchase) |
| Class balance | 84.4% no purchase · 15.6% purchase (after de-duplication; raw file 84.5% / 15.5%) |
| Missing values | None |

**Features**

| Feature | Type | Description |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | numeric | Number of pages of each category visited in the session |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | numeric | Total seconds spent on each page category |
| `BounceRates` | numeric | Avg. % of visitors who enter on the page and leave without another request |
| `ExitRates` | numeric | Avg. % of page views that were the last in the session |
| `PageValues` | numeric | Avg. value of the pages visited before completing a transaction |
| `SpecialDay` | numeric (0–1) | Closeness of the visit date to a special day (e.g. Mother's Day, Valentine's Day) |
| `Month` | categorical | Month of the visit (10 distinct values; Jan and Apr are absent) |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | categorical (integer codes) | Technical / acquisition attributes of the visitor |
| `VisitorType` | categorical | Returning_Visitor / New_Visitor / Other |
| `Weekend` | binary | Whether the visit happened on a weekend |

**Preprocessing (stored inside every saved model pipeline, so the app needs no extra code)**

1. Exact duplicate rows removed; `Weekend` and `Revenue` mapped to 0/1.
2. Stratified 80/20 split → 9,764 training rows and 2,441 test rows (`random_state = 2026`).
   The test rows are saved as [`test_data.csv`](test_data.csv) and never used for training or tuning.
3. Numeric features: `log1p` transform (all ten are heavily right-skewed) followed by `StandardScaler`.
4. `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`: one-hot encoded
   (the integer codes are labels, not quantities). `handle_unknown="ignore"` protects against unseen codes.
5. `Weekend`: passed through as 0/1. Result: 74 model inputs.

## c. GitHub repository link

https://github.com/<your-github-username>/shopper-intent-classifier

```
shopper-intent-classifier/
├── app.py                     # Streamlit app (loads saved models, never retrains)
├── train_models.py            # Preprocessing, GridSearchCV tuning, evaluation, model export
├── requirements.txt           # Pinned dependencies (same versions used to pickle the models)
├── README.md
├── test_data.csv              # 2,441-row hold-out set (17 features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # original UCI file
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics.json           # hold-out metrics + tuned hyper-parameters
│   └── feature_schema.json    # expected columns used by the app for validation
└── .streamlit/config.toml     # app theme
```

## d. Models used

All five models were tuned with 5-fold stratified `GridSearchCV` on the training split, using
F1 as the selection metric (because of the class imbalance). The best pipeline for each model
was then evaluated once on the untouched test split. Predicted labels are
`P(purchase) ≥ 0.5` for every model, which is exactly what the Streamlit app reports at its
default threshold.

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | `C=0.05`, `class_weight="balanced"`, lbfgs solver |
| Decision Tree | `criterion="entropy"`, `max_depth=6`, `min_samples_leaf=30`, `class_weight=None` |
| KNN | `n_neighbors=11`, `weights="uniform"`, Euclidean distance (`p=2`) |
| Naive Bayes (Gaussian) | `var_smoothing=0.1` |
| Support Vector Machine | RBF kernel, `C=0.5`, `gamma=0.01`, `class_weight="balanced"`, Platt-scaled probabilities |

### Comparison table (hold-out test set, n = 2,441)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8804 | 0.9318 | 0.5859 | 0.8037 | **0.6777** | **0.6179** |
| Decision Tree | 0.9013 | **0.9348** | 0.7296 | 0.5864 | 0.6502 | 0.5983 |
| KNN | **0.9033** | 0.9056 | **0.7535** | 0.5681 | 0.6478 | 0.6010 |
| Naive Bayes | 0.7669 | 0.8718 | 0.3888 | **0.8560** | 0.5348 | 0.4636 |
| Support Vector Machine | 0.9009 | 0.9325 | 0.7258 | 0.5890 | 0.6503 | 0.5977 |

Reference point: predicting "no purchase" for every session already gives 0.8435 accuracy
(but 0 recall, 0 F1 and 0 MCC), so accuracy on its own is a weak signal for this dataset.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best F1 (0.678) and best MCC (0.618) of the five, with AUC 0.932 — only 0.003 below the top. Grid search chose strong regularisation (`C=0.05`) and balanced class weights, which shifts the decision boundary toward the minority class: it catches 80% of buyers (recall 0.804) at the price of more false alarms (precision 0.586), so its accuracy (0.880) is lower than the tree/KNN/SVM. The log1p transform makes the dominant signal (`PageValues`, largest coefficient +1.53) close to linear in log-odds, which is why a simple linear model works so well. `Month_Nov` (+0.97) and `Month_May` (−0.77) are the next strongest effects, i.e. seasonality matters. |
| Decision Tree | Highest AUC (0.935) and second-best accuracy (0.901). The root split is `PageValues`, which alone carries 72% of the tree's feature importance, followed by `Month_Nov` and `BounceRates`. Depth was limited to 6 with ≥30 samples per leaf; deeper trees overfit in cross-validation. Without class weighting it is conservative: precision 0.730 but it misses ~41% of buyers (recall 0.586). Its ranking is good, so lowering the threshold to 0.3 raises recall to 0.715 and F1 to 0.682. Very fast and fully interpretable. |
| KNN | Highest accuracy (0.903) and highest precision (0.754), but the lowest recall of all five models (0.568) and the second-lowest AUC (0.906). With 74 inputs — most of them sparse one-hot columns — Euclidean distances are dominated by the majority class, so a buyer's 11 neighbours are often mostly non-buyers. Its probability estimates are coarse (multiples of 1/11), which limits AUC. It also has to store the whole training set (largest model file, 566 KB). |
| Naive Bayes | Weakest overall: accuracy 0.767, precision 0.389, F1 0.535, MCC 0.464. It labels many sessions as buyers, giving the highest recall (0.856) but many false positives. The feature-independence assumption is clearly violated here (e.g. `BounceRates`–`ExitRates` correlation 0.91, `ProductRelated`–`ProductRelated_Duration` 0.86), so the same evidence is counted several times, and the Gaussian assumption is poor for 0/1 one-hot columns. A large `var_smoothing` (0.1) was needed to stabilise it. Still useful as a fast, high-recall baseline. |
| Support Vector Machine | Best cross-validated F1 during tuning (0.665) and a strong AUC (0.933), so it ranks sessions as well as the best models. At the 0.5 probability threshold, however, it behaves like the unweighted models (precision 0.726, recall 0.589, F1 0.650, MCC 0.598): Platt scaling re-calibrates the probabilities and removes most of the shift that `class_weight="balanced"` gave the raw decision function. Lowering the threshold to 0.3 gives recall 0.736, F1 0.683, MCC 0.621. It was by far the slowest to train (~1 min for the grid vs. a few seconds for the others). |
| Overall, Winner for your dataset? | **Logistic Regression.** At the standard 0.5 threshold it has the best F1 (0.678) and the best MCC (0.618), both of which account for the 84/16 class imbalance, and it finds the most buyers among the models with acceptable precision. Its AUC (0.932) is within 0.003 of the Decision Tree and SVM, and it is the cheapest and most interpretable model. If ranking quality (AUC) is all that matters, the Decision Tree is marginally best; if raw accuracy is the target, KNN wins — but neither of those is a good criterion on an imbalanced dataset. The SVM and Decision Tree can match Logistic Regression only after the threshold is tuned. |

---

## Streamlit app

Features (all evaluated on the uploaded file — models are loaded from `model/`, never retrained):

- **CSV upload** of test data (must contain the 17 feature columns and `Revenue`; `TRUE/FALSE`,
  `True/False` and `1/0` are all accepted). The bundled `test_data.csv` can be downloaded from the
  sidebar or used directly with one click.
- **Model selection dropdown** for the five classifiers, with the tuned hyper-parameters shown.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC.
- **Confusion matrix** (counts and %) and **classification report**, plus the ROC curve.
- **Compare all models** tab: every model on the same uploaded data, with a metric bar chart and overlaid ROC curves.
- **Predictions** tab: per-row purchase probability and predicted label, filter for mistakes, download as CSV.
- **Decision-threshold slider** to explore the precision/recall trade-off (0.50 reproduces the table above).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_models.py               # optional: re-creates model/*.joblib, metrics.json and test_data.csv
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → select this repository, branch `main`, main file `app.py`.
4. Under *Advanced settings* choose Python **3.11** or **3.12** (the pinned packages support 3.9–3.13).
5. Click **Deploy**.

`requirements.txt` pins `scikit-learn==1.6.1`, the version used to save the models; a different
scikit-learn version may fail to load the `.joblib` files.
