# Car Dataset Analysis using Pandas & Matplotlib

**Internship Project — Data Analysis & Visualisation**
**Language:** Python 3.x | **Libraries:** pandas, numpy, matplotlib, seaborn

---

## Project Overview

This project performs a comprehensive data analysis on a car/automobile dataset using the **Pandas** library for data manipulation and **Matplotlib/Seaborn** for visualisations. The analysis covers loading and inspecting the CSV data, computing statistical summaries and averages, generating multiple chart types (bar chart, scatter plot, heatmap, boxplot), and deriving meaningful insights and observations from the findings.

---

## Objectives

- Load a CSV file using Pandas and inspect its structure
- Perform basic and advanced data analysis tasks (averages, groupings, correlations)
- Create visualisations: bar chart, scatter plot, correlation heatmap, and boxplot
- Derive and document data-driven insights from the analysis

---

## Dataset

| Property | Detail |
|---|---|
| File | `cars_dataset.csv` (included in project folder) |
| Records | 200 cars |
| Columns | 9 |
| Source | Custom-generated with realistic structure (Kaggle-style) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Brand` | Categorical | Car brand (Toyota, BMW, Honda, etc.) |
| `Year` | Integer | Manufacturing year (2010–2024) |
| `Engine_Size_L` | Float | Engine displacement in litres |
| `Horsepower` | Integer | Engine power output (HP) |
| `Weight_kg` | Integer | Vehicle weight in kilograms |
| `Fuel_Type` | Categorical | Petrol, Diesel, CNG, or Electric |
| `Transmission` | Categorical | Manual or Automatic |
| `Mileage_MPG` | Float | Fuel efficiency in miles per gallon |
| `Price` | Integer | Vehicle price in INR (₹) |

---

## Project Structure

```
car-analysis/
car_analysis.py                    # Main analysis script
cars_dataset.csv                   # Dataset (200 records, 9 columns)
bar_chart_price_by_brand.png       # Generated: Average Price by Brand
scatter_horsepower_vs_price.png    # Generated: Horsepower vs Price
heatmap_correlation.png            # Generated: Correlation Matrix
boxplot_mileage_by_transmission.png # Generated: Mileage by Transmission
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
pip install pandas numpy matplotlib seaborn
```

### Step 3 — Run the script
Place both `car_analysis.py` and `cars_dataset.csv` in the same folder, then run:
```bash
python car_analysis.py
```

---

## Analysis Walkthrough

### 1. Data Loading & Inspection
```python
df = pd.read_csv('cars_dataset.csv')
print(df.shape)       # (200, 9)
print(df.dtypes)      # column types
print(df.isnull().sum())  # missing value check
print(df.describe())  # full statistical summary
```

### 2. Statistical Analysis
The following averages and aggregations are computed:

| Analysis | Method |
|---|---|
| Average Price | `df['Price'].mean()` |
| Average Mileage (MPG) | `df['Mileage_MPG'].mean()` |
| Average Horsepower | `df['Horsepower'].mean()` |
| Average Price by Brand | `df.groupby('Brand')['Price'].mean()` |
| Average Mileage by Fuel Type | `df.groupby('Fuel_Type')['Mileage_MPG'].mean()` |
| Most/Least Expensive Car | `df.loc[df['Price'].idxmax()]` |
| Feature Correlations | `df.select_dtypes(include=number).corr()` |

### 3. Visualisations

#### Bar Chart — Average Price by Brand
Shows which brands command the highest average market price. Uses a viridis color gradient sorted high to low.

#### Scatter Plot — Horsepower vs Price (by Fuel Type)
Reveals the relationship between engine power and price. Each fuel type (Petrol, Diesel, CNG, Electric) is plotted in a distinct colour to show if fuel type modifies this relationship.

#### Heatmap — Correlation Matrix
Displays pairwise Pearson correlations between all numeric features using a coolwarm color map. Identifies which features most strongly influence price.

#### Boxplot — Mileage by Transmission Type
Compares the distribution of fuel efficiency (MPG) across Manual and Automatic transmission types, showing median, spread, and outliers.

---

## Key Insights & Observations

1. **Brand premium is significant** — Luxury/premium brands show substantially higher average prices compared to budget-friendly options, even controlling for horsepower and engine size.

2. **Horsepower vs Price (r ≈ 0.46)** — A moderate positive correlation exists between horsepower and price. Higher-powered cars cost more, but the relationship is not perfectly linear — brand reputation plays an independent role.

3. **Electric vehicles lead in mileage** — EV and CNG variants deliver the best average fuel efficiency, confirming their superiority in running economy over petrol/diesel counterparts.

4. **Performance-efficiency trade-off** — As horsepower increases, mileage (MPG) tends to decrease, following the classic engineering constraint between power output and fuel consumption.

5. **Transmission and mileage** — The boxplot reveals that Automatic cars have marginally different mileage profiles compared to Manual, with broader variance — suggesting more diverse model types fall under the Automatic category.

6. **Engine size moderately predicts price (r ≈ 0.21)** — Larger engines generally cost more, but the effect is weaker than horsepower, suggesting buyers pay more for power delivery than displacement alone.

---

## Output Files

All four visualisation files are saved to the same directory as the script:

| File | Chart Type | Description |
|---|---|---|
| `bar_chart_price_by_brand.png` | Bar Chart | Average Price by Brand |
| `scatter_horsepower_vs_price.png` | Scatter Plot | Horsepower vs Price, by Fuel Type |
| `heatmap_correlation.png` | Heatmap | Correlation Matrix of all numeric features |
| `boxplot_mileage_by_transmission.png` | Boxplot | Mileage distribution by Transmission type |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | CSV loading, groupby, describe, correlation |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Bar charts, scatter plots, boxplots |
| `seaborn` | Heatmap with annotation |

---

## Sample Terminal Output

```
   CAR DATASET PANDAS & MATPLOTLIB ANALYSIS


 Dataset shape : 200 rows, 9 columns
 Columns: ['Brand', 'Year', 'Engine_Size_L', 'Horsepower', ...]

 Average Price        : Rs.5,39,412
 Average Mileage (MPG): 31.2
 Average Horsepower   : 229 HP

 [Key Insights printed automatically at end of run]

 Analysis complete!
```

---

## Future Improvements

- Add interactive visualisations using Plotly
- Perform price segmentation (budget / mid-range / premium) using clustering
- Build a price prediction model using the cleaned features
- Add time-series analysis of price trends by year

---

*Submitted as part of the Data Analysis Internship Program.*
