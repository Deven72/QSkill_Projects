# House Price Prediction using Linear Regression

**Internship Project — Machine Learning**
**Language:** Python 3.x | **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn

---

## Project Overview

This project builds a supervised machine learning pipeline that predicts residential house sale prices based on structural and locational features such as living area, overall quality, garage capacity, neighbourhood, and more. The dataset is sourced from the **Kaggle House Prices: Advanced Regression Techniques** competition (Ames Housing dataset — `train.csv`), which contains 1,460 real records across 81 columns.

Two regression models are trained and compared — **Ordinary Least Squares (OLS) Linear Regression** and **Ridge Regression (L2 regularisation)** — with full evaluation using MAE, RMSE, R², MAPE, and 5-fold cross-validation.

---

## Objectives

- Load and explore a real-world housing dataset from Kaggle
- Perform Exploratory Data Analysis (EDA) to discover patterns and correlations
- Preprocess data — handle missing values, encode categoricals, engineer features
- Train and evaluate Linear Regression and Ridge Regression models
- Interpret model coefficients and visualise prediction accuracy

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| File | `train.csv` |
| Records | 1,460 |
| Original Columns | 81 |
| Target Variable | `SalePrice` (residential property sale price in USD) |
| Features Used | 11 (selected + engineered) |

### Features Used After Engineering

| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality (1–10) |
| `GrLivArea` | Above-grade (ground) living area in sq ft |
| `TotalBsmtSF` | Total basement area in sq ft |
| `GarageCars` | Garage capacity in car count |
| `FullBath` | Full bathrooms above grade |
| `BedroomAbvGr` | Bedrooms above grade |
| `HouseAge` | 2024 − YearBuilt (engineered) |
| `LotArea` | Lot size in sq ft |
| `Neighborhood_enc` | Neighbourhood label-encoded (engineered) |
| `TotalSF` | GrLivArea + TotalBsmtSF (engineered) |
| `QualArea` | OverallQual × GrLivArea — interaction term (engineered) |

---

## Project Structure

```
house-price-prediction/
house.py           # Main ML pipeline script
train.csv          # Kaggle dataset (download separately)
eda_plots.png      # Generated: 6-panel EDA visualisation
model_results.png  # Generated: Actual vs Predicted + Feature Importance
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1 — Create a virtual environment
```bash
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 3 — Download the dataset
1. Go to [kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. Download `train.csv`
3. Place it in the same folder as `house.py`

### Step 4 — Run the script
```bash
python house.py
```

---

## Pipeline Walkthrough

### 1. Data Loading
```python
df = pd.read_csv('train.csv')
```
Loads 1,460 records. The script selects the 11 most predictive columns, then handles any missing values.

### 2. Exploratory Data Analysis
Six plots are generated and saved to `eda_plots.png`:
- Sale price distribution (histogram)
- Price vs Living Area (scatter)
- Median price by Overall Quality (bar chart)
- Median price by Neighbourhood (bar chart)
- Correlation heatmap of all numeric features
- Price vs Year Built (scatter)

### 3. Preprocessing
| Step | Technique |
|---|---|
| Missing values | `SimpleImputer(strategy='median')` |
| Categorical encoding | `LabelEncoder` on Neighbourhood |
| Feature engineering | HouseAge, TotalSF, QualArea |
| Scaling | `StandardScaler` (fit on train, transform on test) |
| Train/Test split | 80% train / 20% test, `random_state=42` |

### 4. Model Training
```python
lr  = LinearRegression()
rid = Ridge(alpha=10)

lr.fit(X_train_sc, y_train)
rid.fit(X_train_sc, y_train)
```

### 5. Evaluation
Both models are evaluated using:
- **MAE** — Mean Absolute Error (avg. dollar prediction error)
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **R^2** — Coefficient of determination (variance explained)
- **MAPE** — Mean Absolute Percentage Error
- **5-fold Cross-Validation R^2** — Generalisation estimate

---

## Results

| Metric | Linear Regression | Ridge Regression |
|---|---|---|
| MAE | $23,199 | $23,357 |
| RMSE | $36,327 | $36,696 |
| **R^2 Score** | **0.8280** | 0.8244 |
| MAPE | 13.81% | 13.86% |
| CV R^2 (5-fold) | 0.7672 ± 0.09 | 0.7675 ± 0.09 |

**Linear Regression is the better model**, explaining **82.8% of price variance** with an average prediction error of ~$23K — a strong result for a linear model on real-world, noisy housing data.

### Key Insight — Feature Importance
Based on absolute scaled coefficients, the strongest price predictors are:
1. **QualArea** (OverallQual × GrLivArea) — interaction of quality and size
2. **OverallQual** — overall material and finish quality rating
3. **GrLivArea / TotalSF** — total living space
4. **Neighbourhood** — location remains a key driver
5. **GarageCars** and **FullBath** — amenity value

---

## Output Files

After running the script, two image files are saved in the same directory:

| File | Contents |
|---|---|
| `eda_plots.png` | 6-panel EDA — distribution, scatter plots, bar charts, heatmap |
| `model_results.png` | Actual vs Predicted (LR + Ridge) and Feature Importance bar chart |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, filtering, grouping |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Heatmap and advanced chart styling |
| `scikit-learn` | Preprocessing, model training, evaluation |

---

## Future Improvements

- Log-transform `SalePrice` to reduce right skew and improve RMSE
- Use Polynomial or Interaction features for non-linear relationships
- Try ensemble methods: Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning via GridSearchCV
- Deploy as a REST API using Flask or FastAPI

---

*Submitted as part of the Machine Learning Internship Program.*
