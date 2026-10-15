# Data Dictionary – Breast Cancer Wisconsin (Diagnostic)

* **File:** `breast_cancer_wisconsin.csv` (569 rows × 31 columns)
* **Source:** UCI Machine Learning Repository – *Breast Cancer Wisconsin (Diagnostic)* (Wolberg, Street & Mangasarian, 1993).
  Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
  (identical copy is bundled with scikit-learn: `sklearn.datasets.load_breast_cancer`).
* **Collection:** features are computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass and describe the cell nuclei present.

## Target
| Column | Meaning |
|---|---|
| `diagnosis` | `M` = malignant (212 cases), `B` = benign (357 cases) |

## Features (30) – 10 measurements × 3 statistics
Ten real-valued measurements are computed for every nucleus, then summarised per image as **mean**, **standard error** and **worst** (mean of the three largest values).

| Base measurement | Meaning | Columns |
|---|---|---|
| radius | mean distance from centre to points on the perimeter | `mean_radius`, `radius_error`, `worst_radius` |
| texture | standard deviation of grey-scale values | `mean_texture`, `texture_error`, `worst_texture` |
| perimeter | nucleus perimeter | `mean_perimeter`, `perimeter_error`, `worst_perimeter` |
| area | nucleus area | `mean_area`, `area_error`, `worst_area` |
| smoothness | local variation in radius lengths | `mean_smoothness`, `smoothness_error`, `worst_smoothness` |
| compactness | perimeter² / area − 1.0 | `mean_compactness`, `compactness_error`, `worst_compactness` |
| concavity | severity of concave portions of the contour | `mean_concavity`, `concavity_error`, `worst_concavity` |
| concave_points | number of concave portions of the contour | `mean_concave_points`, `concave_points_error`, `worst_concave_points` |
| symmetry | symmetry of the nucleus | `mean_symmetry`, `symmetry_error`, `worst_symmetry` |
| fractal_dimension | "coastline approximation" − 1 | `mean_fractal_dimension`, `fractal_dimension_error`, `worst_fractal_dimension` |

## Processed splits (`data/processed/`)
`train.csv` (398), `val.csv` (85), `test.csv` (86): Yeo-Johnson-transformed (22 skewed features) and z-score standardised with a scaler fitted on the training set only. Column `diagnosis` is encoded **1 = malignant, 0 = benign**.
