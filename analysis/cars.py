import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. LOAD CSV WITH PANDAS

df = pd.read_csv('cars_dataset.csv')
print("=" * 55)
print("   CAR DATASET — PANDAS & MATPLOTLIB ANALYSIS")
print("=" * 55)

print(f"\n Dataset shape : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n Columns: {list(df.columns)}")
print(f"\n First 5 rows:\n{df.head()}")
print(f"\n Data types:\n{df.dtypes}")
print(f"\n Missing values:\n{df.isnull().sum()}")

 
# 2. BASIC DATA ANALYSIS

print("\n" + "─" * 55)
print(" BASIC STATISTICS")
print("─" * 55)

print(f"\n Full statistical summary:\n{df.describe()}")

# Average of selected columns
avg_price = df['Price'].mean()
avg_mpg = df['Mileage_MPG'].mean()
avg_hp = df['Horsepower'].mean()

print(f"\n Average Price        : ₹{avg_price:,.0f}")
print(f" Average Mileage (MPG): {avg_mpg:.2f}")
print(f" Average Horsepower   : {avg_hp:.1f} HP")

# Average price by brand
avg_price_by_brand = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
print(f"\n Average Price by Brand:\n{avg_price_by_brand.round(0)}")

# Average mileage by fuel type
avg_mpg_by_fuel = df.groupby('Fuel_Type')['Mileage_MPG'].mean().sort_values(ascending=False)
print(f"\n Average Mileage by Fuel Type:\n{avg_mpg_by_fuel.round(2)}")

# Most expensive & cheapest car
most_expensive = df.loc[df['Price'].idxmax()]
cheapest = df.loc[df['Price'].idxmin()]
print(f"\n Most Expensive Car : {most_expensive['Brand']} ({most_expensive['Year']}) - ₹{most_expensive['Price']:,}")
print(f" Cheapest Car       : {cheapest['Brand']} ({cheapest['Year']}) - ₹{cheapest['Price']:,}")

# Correlation between numeric features
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
print(f"\n Correlation with Price:\n{correlation['Price'].sort_values(ascending=False)}")


# 3. VISUALIZATIONS

plt.style.use('seaborn-v0_8-whitegrid')

#3.1 BAR CHART — Average Price by Brand
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_price_by_brand)))
plt.bar(avg_price_by_brand.index, avg_price_by_brand.values, color=colors)
plt.title('Average Car Price by Brand', fontsize=14, fontweight='bold')
plt.xlabel('Brand')
plt.ylabel('Average Price (Rs.)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bar_chart_price_by_brand.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n [✓] Bar chart saved → bar_chart_price_by_brand.png")

#3.2 SCATTER PLOT — Horsepower vs Price 
plt.figure(figsize=(9, 6))
fuel_colors = {'Petrol': 'tomato', 'Diesel': 'steelblue', 'CNG': 'mediumseagreen', 'Electric': 'gold'}
for fuel, color in fuel_colors.items():
    subset = df[df['Fuel_Type'] == fuel]
    plt.scatter(subset['Horsepower'], subset['Price'], label=fuel,
                alpha=0.6, color=color, s=50, edgecolors='white')
plt.title('Horsepower vs Price (colored by Fuel Type)', fontsize=14, fontweight='bold')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Price (Rs.)')
plt.legend(title='Fuel Type')
plt.tight_layout()
plt.savefig('scatter_horsepower_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print(" [✓] Scatter plot saved → scatter_horsepower_vs_price.png")

#3.3 HEATMAP — Correlation Matrix
plt.figure(figsize=(9, 7))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap — Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(" [✓] Heatmap saved → heatmap_correlation.png")

#3.4 BONUS — Mileage Distribution by Transmission (boxplot)
plt.figure(figsize=(8, 6))
df.boxplot(column='Mileage_MPG', by='Transmission', grid=False,
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
plt.title('Mileage Distribution by Transmission Type')
plt.suptitle('')
plt.xlabel('Transmission')
plt.ylabel('Mileage (MPG)')
plt.tight_layout()
plt.savefig('boxplot_mileage_by_transmission.png', dpi=150, bbox_inches='tight')
plt.close()
print(" [✓] Boxplot saved → boxplot_mileage_by_transmission.png")


# 4. INSIGHTS & OBSERVATIONS

print("\n" + "═" * 55)
print(" KEY INSIGHTS & OBSERVATIONS")
print("═" * 55)

top_brand = avg_price_by_brand.idxmax()
cheap_brand = avg_price_by_brand.idxmin()
best_mileage_fuel = avg_mpg_by_fuel.idxmax()
hp_price_corr = correlation.loc['Horsepower', 'Price']
auto_mileage = df[df['Transmission'] == 'Automatic']['Mileage_MPG'].mean()
manual_mileage = df[df['Transmission'] == 'Manual']['Mileage_MPG'].mean()

print(f"""
 1. {top_brand} has the highest average price (₹{avg_price_by_brand.max():,.0f}),
    while {cheap_brand} is the most budget-friendly (₹{avg_price_by_brand.min():,.0f}).

 2. Horsepower and Price show a {'strong' if abs(hp_price_corr)>0.5 else 'moderate'} positive
    correlation (r = {hp_price_corr:.2f}) — higher-powered cars tend to cost more.

 3. {best_mileage_fuel} vehicles deliver the best average mileage
    ({avg_mpg_by_fuel.max():.1f} MPG) among all fuel types.

 4. Manual transmission cars average {manual_mileage:.1f} MPG vs
    {auto_mileage:.1f} MPG for Automatic — {'manual' if manual_mileage>auto_mileage else 'automatic'}
    vehicles are more fuel-efficient in this dataset.

 5. Mileage tends to decrease as horsepower increases, confirming the classic
    performance-vs-efficiency trade-off in vehicle design.

 6. Engine size and price are positively correlated (r = {correlation.loc['Engine_Size_L','Price']:.2f}),
    suggesting engine capacity is a meaningful price driver alongside brand and power.
""")

print(" Analysis complete! Check the generated PNG files for visualizations.\n")