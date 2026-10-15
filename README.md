# 🔬 EDA & Visualization of the Breast Cancer Wisconsin (Diagnostic) Dataset

**Course:** U21ADP05 – Exploratory Data Analysis and Visualization · Assignment II (Case Study)  
**Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology**  
**Student:** `<YOUR NAME>` · **Roll No.:** `<YOUR ROLL NO>` · **Batch:** 2023–2027 · **Year/Sem:** III / V

---

## 📌 Project summary
An end-to-end exploratory data analysis, statistical analysis, visualization and deep-learning case study on the
**Breast Cancer Wisconsin (Diagnostic)** dataset (UCI) – 569 patients, **30 numeric features** computed from fine-needle-aspirate
images, with a benign/malignant diagnosis. The project follows Ben Fry's **7 stages of visualization**
(Acquire → Parse → Filter → Mine → Represent → Refine → Interact) and finishes with a multi-layer perceptron (deep neural network)
that is tuned, evaluated and compared with two baselines.

## 🎯 Key results
| Item | Result |
|---|---|
| Dataset | 569 rows × 30 features (212 malignant / 357 benign), no missing values or duplicates |
| Visualizations | 17 figures + 1 screenshot (distribution, box, outlier, correlation, scatter, pair-plot, Q-Q/PDF fits, binomial, PCA, tuning, loss/accuracy, confusion matrix, ROC/PR, error distribution, feature importance) + 1 interactive HTML explorer |
| Model | MLP 30 → 64 → 32 → 16 → 1 (ReLU, Adam, L2, early stopping), 4,609 parameters |
| Test accuracy / recall / AUC | **98.8 % / 96.9 % / ≈1.00** (86 unseen patients, 1 error) |
| 5-fold CV (train+val) | accuracy 96.5 % ± 3.0 %, AUC 0.992 ± 0.006 |

## 📁 Repository structure
```
breast-cancer-eda-visualization/
├── README.md                     ← you are here
├── requirements.txt              ← Python dependencies
├── .gitignore
├── data/
│   ├── breast_cancer_wisconsin.csv   ← raw dataset (UCI WDBC)
│   ├── DATA_DICTIONARY.md            ← description of every field + source link
│   └── processed/                    ← train.csv · val.csv · test.csv (cleaned, scaled, split)
├── notebooks/
│   └── EDA_Visualization_MLP.ipynb   ← full executed notebook (outputs & charts included)
├── src/
│   └── eda_mlp_pipeline.py           ← same pipeline as a runnable script
├── figures/                      ← 17 figures + interactive-explorer screenshot
├── interactive/
│   └── scatter_explorer.html         ← open in a browser (Stage 7: Interact)
├── outputs/                      ← metrics.json, tuning results, test results, effect sizes, distribution fits
└── reports/
    ├── Project_Report.docx           ← full project report (Word)
    └── Project_Report.pdf            ← same report (PDF)
```

## ▶️ How to run
```bash
git clone https://github.com/<your-username>/breast-cancer-eda-visualization.git
cd breast-cancer-eda-visualization
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A – notebook
jupyter notebook notebooks/EDA_Visualization_MLP.ipynb     # Run → Restart & Run All

# Option B – script (regenerates all figures + outputs)
python src/eda_mlp_pipeline.py
```
Everything is seeded (`SEED = 42`) so results are reproducible.

## 🧭 Pipeline (7 stages of visualization)
| Stage | What was done |
|---|---|
| 1 Acquire | Loaded UCI WDBC data (CSV in `data/`) |
| 2 Parse | Data types, shape, feature groups (mean / error / worst), summary statistics |
| 3 Filter | Missing values, duplicates, IQR outliers (kept – clinically genuine), zero-variance check |
| 4 Mine | Skewness, Mann-Whitney U & Cohen's d, Normal/Log-normal fits (AIC, KS), Binomial model, correlation, PCA |
| 5 Represent | 17 visualizations with *what / why / insight* explained in the report |
| 6 Refine | Stratified 70/15/15 split, Yeo-Johnson power transform for 22 skewed features, z-score scaling (both fitted on train only), consistent colour-blind-safe style |
| 7 Interact | `interactive/scatter_explorer.html` – choose any 2 features, filter classes, hover for details |

## 🧠 Model
MLP (feed-forward deep neural network): `30 → 64 → 32 → 16 → 1`, ReLU hidden units, sigmoid output, Adam optimiser, binary cross-entropy,
mini-batch 32, L2 penalty 0.01, learning rate 0.001, early stopping on validation loss (best epoch 12).
Grid search over 24 configurations (4 architectures × 2 L2 penalties × 3 learning rates) on the validation set.
Baselines: Logistic Regression and Random Forest. Implemented with **scikit-learn's `MLPClassifier`**.

## ⚠️ Limitations
* Small dataset (569 patients) → the 86-patient test set is noisy; cross-validation is the more reliable estimate.
* Features are pre-extracted measurements, not raw images; results are for education only and **not** a clinical tool.

## 📚 References
See `reports/Project_Report.pdf` (Section 8) – UCI repository, Street et al. (1993), Fry (2007), Tukey (1977), Pedregosa et al. (2011), Goodfellow et al. (2016).
