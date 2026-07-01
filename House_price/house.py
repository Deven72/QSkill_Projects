import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# 1. DATASET CREATION (Kaggle-style structure)

np.random.seed(42)
n = 1000

neighborhoods = ['Downtown', 'Suburbs', 'Rural', 'Uptown', 'Midtown']
neighborhood_arr = np.random.choice(neighborhoods, n)
neighborhood_price = {'Downtown': 50000, 'Uptown': 40000, 'Midtown': 30000,
                      'Suburbs': 15000, 'Rural': 0}

overall_qual = np.random.randint(1, 11, n)
gr_liv_area  = np.random.randint(500, 4000, n)
total_bsmt_sf= np.random.randint(0, 2000, n)
garage_cars  = np.random.randint(0, 4, n)
full_bath    = np.random.randint(0, 4, n)
bedroom_abvgr= np.random.randint(1, 6, n)
year_built   = np.random.randint(1900, 2023, n)
lot_area     = np.random.randint(2000, 20000, n)

# Realistic price formula with noise
price = (
    overall_qual * 12000 +
    gr_liv_area  * 60    +
    total_bsmt_sf* 25    +
    garage_cars  * 8000  +
    full_bath    * 5000  +
    bedroom_abvgr* 3000  +
    (2023 - year_built) * (-300) +
    lot_area     * 2     +
    np.array([neighborhood_price[n_] for n_ in neighborhood_arr]) +
    np.random.normal(0, 15000, n)
)
price = np.clip(price, 50000, 800000)

df = pd.DataFrame({
    'OverallQual':   overall_qual,
    'GrLivArea':     gr_liv_area,
    'TotalBsmtSF':   total_bsmt_sf,
    'GarageCars':    garage_cars,
    'FullBath':      full_bath,
    'BedroomAbvGr':  bedroom_abvgr,
    'YearBuilt':     year_built,
    'LotArea':       lot_area,
    'Neighborhood':  neighborhood_arr,
    'SalePrice':     price.astype(int)
})

# missing values to simulate real Kaggle data
for col in ['GrLivArea', 'TotalBsmtSF', 'GarageCars']:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan

print("=" * 55)
print("   HOUSE PRICE PREDICTION — LINEAR REGRESSION")
print("=" * 55)
print(f"\n Dataset shape : {df.shape}")
print(f" Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")


# 2. EXPLORATORY DATA ANALYSIS  (save plots)

plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — House Price Dataset', fontsize=16, fontweight='bold')

# Price distribution
axes[0,0].hist(df['SalePrice'], bins=40, color='steelblue', edgecolor='white')
axes[0,0].set_title('Sale Price Distribution')
axes[0,0].set_xlabel('Sale Price ($)')
axes[0,0].set_ylabel('Frequency')

# Price vs GrLivArea
axes[0,1].scatter(df['GrLivArea'], df['SalePrice'], alpha=0.4, color='coral')
axes[0,1].set_title('Price vs Living Area')
axes[0,1].set_xlabel('Above-grade Living Area (sq ft)')
axes[0,1].set_ylabel('Sale Price ($)')

# Price by Overall Quality
qual_price = df.groupby('OverallQual')['SalePrice'].median()
axes[0,2].bar(qual_price.index, qual_price.values, color='mediumseagreen')
axes[0,2].set_title('Median Price by Overall Quality')
axes[0,2].set_xlabel('Overall Quality (1–10)')
axes[0,2].set_ylabel('Median Sale Price ($)')

# Price by Neighborhood
nb_price = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
axes[1,0].bar(nb_price.index, nb_price.values, color='mediumpurple')
axes[1,0].set_title('Median Price by Neighborhood')
axes[1,0].set_xlabel('Neighborhood')
axes[1,0].set_ylabel('Median Sale Price ($)')
axes[1,0].tick_params(axis='x', rotation=20)

# Correlation heatmap
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,1], annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, annot_kws={'size': 7})
axes[1,1].set_title('Feature Correlation Matrix')

# Price vs Year Built
axes[1,2].scatter(df['YearBuilt'], df['SalePrice'], alpha=0.3, color='goldenrod')
axes[1,2].set_title('Price vs Year Built')
axes[1,2].set_xlabel('Year Built')
axes[1,2].set_ylabel('Sale Price ($)')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n [✓] EDA plots saved")


# 3. PREPROCESSING

df_model = df.copy()

# Encode categorical
le = LabelEncoder()
df_model['Neighborhood_enc'] = le.fit_transform(df_model['Neighborhood'])

# Feature engineering
df_model['HouseAge']      = 2023 - df_model['YearBuilt']
df_model['TotalSF']       = df_model['GrLivArea'].fillna(0) + df_model['TotalBsmtSF'].fillna(0)
df_model['QualArea']      = df_model['OverallQual'] * df_model['GrLivArea']

features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
            'FullBath', 'BedroomAbvGr', 'HouseAge', 'LotArea',
            'Neighborhood_enc', 'TotalSF', 'QualArea']

X = df_model[features]
y = df_model['SalePrice']

# Impute missing
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)
X_imp = pd.DataFrame(X_imp, columns=features)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n Training samples : {X_train.shape[0]}")
print(f" Testing  samples : {X_test.shape[0]}")
print(f" Features used    : {len(features)}")

# 4. MODEL TRAINING

lr  = LinearRegression()
rid = Ridge(alpha=10)

lr.fit(X_train_sc, y_train)
rid.fit(X_train_sc, y_train)

y_pred_lr  = lr.predict(X_test_sc)
y_pred_rid = rid.predict(X_test_sc)

def metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n [{name}]")
    print(f"MAE  : ${mae:,.0f}")
    print(f"RMSE : ${rmse:,.0f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

print("\n" + "─"*55)
print(" MODEL EVALUATION RESULTS")
print("─"*55)
m_lr  = metrics(y_test, y_pred_lr,  "Linear Regression")
m_rid = metrics(y_test, y_pred_rid, "Ridge Regression")

# Cross-validation
cv_lr  = cross_val_score(lr,  X_imp, y, cv=5, scoring='r2')
cv_rid = cross_val_score(rid, X_imp, y, cv=5, scoring='r2')
print(f"\n [Cross-Val R² — Linear] : {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")
print(f" [Cross-Val R² — Ridge]  : {cv_rid.mean():.4f} ± {cv_rid.std():.4f}")


# 5. RESULT PLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Results — Linear Regression vs Ridge', fontsize=14, fontweight='bold')

# Actual vs Predicted (LR)
axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color='steelblue', s=20)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
axes[0].set_title(f'Linear Regression\nR² = {m_lr["R2"]:.4f}')
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].legend()

# Actual vs Predicted (Ridge)
axes[1].scatter(y_test, y_pred_rid, alpha=0.4, color='mediumseagreen', s=20)
axes[1].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
axes[1].set_title(f'Ridge Regression\nR² = {m_rid["R2"]:.4f}')
axes[1].set_xlabel('Actual Price ($)')
axes[1].set_ylabel('Predicted Price ($)')
axes[1].legend()

# Feature Importance
coef_df = pd.Series(np.abs(lr.coef_), index=features).sort_values(ascending=True)
axes[2].barh(coef_df.index, coef_df.values, color='coral')
axes[2].set_title('Feature Importance (|Coefficients|)')
axes[2].set_xlabel('Absolute Coefficient Value')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n Model result plots saved")


# 6. SAMPLE PREDICTIONS

sample = pd.DataFrame([{
    'OverallQual': 7, 'GrLivArea': 1800, 'TotalBsmtSF': 900,
    'GarageCars': 2, 'FullBath': 2, 'BedroomAbvGr': 3,
    'HouseAge': 20, 'LotArea': 8000, 'Neighborhood_enc': 0,
    'TotalSF': 2700, 'QualArea': 7*1800
}])
sample_sc = scaler.transform(imputer.transform(sample))
pred_price = lr.predict(sample_sc)[0]
print(f"\n [Sample Prediction]")
print(f"   Features : 7-quality, 1800sqft, 2-car garage, 3bed/2bath, 20yr old")
print(f"   Predicted Price: ${pred_price:,.0f}")
print("\n All done!\n")

# Store metrics for PDF report
METRICS = {'lr': m_lr, 'rid': m_rid,
           'cv_lr': cv_lr, 'cv_rid': cv_rid,
           'sample_pred': pred_price,
           'n_train': X_train.shape[0],
           'n_test': X_test.shape[0],
           'features': features}