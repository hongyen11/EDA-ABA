import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv(r'C:\Users\notco\Documents\EDA Superstore\cleaned_superstore.csv', encoding='latin1')

# Keep leading zeros in Postal Code
df['Postal Code'] = df['Postal Code'].astype(str).str.zfill(5)

# Drop unnecessary columns
df.drop(['Row ID', 'Order ID', 'Product ID'], axis=1, inplace=True, errors='ignore')

# Convert to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

# Feature engineering
df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month_name()
df['Profit Margin'] = df['Profit'] / df['Sales']

# Output directory
output_dir = r'C:\Users\notco\Documents\EDA CONFIRMED'
os.makedirs(output_dir, exist_ok=True)

# Create descriptive summary without Postal Code
desc_summary = df.drop(columns=['Postal Code'], errors='ignore').select_dtypes(include=[np.number]).describe()

# DESCRIPTIVE SUMMARY
print("Dataset Overview:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())
print("\nMode:")
print(df.mode(numeric_only=True))



# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Drop unwanted columns
exclude_cols = ['Postal Code']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Create descriptive summary
desc_summary = df[numeric_cols].describe()

# Plot table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=desc_summary.round(2).values,
                 colLabels=desc_summary.columns,
                 rowLabels=desc_summary.index,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

# Save as PNG
plt.savefig(os.path.join(output_dir, 'descriptive_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# SUPPLY CHAIN ANALYSIS
shipmode_duration = df.groupby('Ship Mode')['Shipping Duration'].mean().sort_values()
shipmode_duration.plot(kind='bar', title='Avg Shipping Duration by Ship Mode', ylabel='Days', xlabel='Ship Mode')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'avg_ship_duration_shipmode.png'))
plt.close()

region_duration = df.groupby('Region')['Shipping Duration'].mean().sort_values()
region_duration.plot(kind='bar', title='Avg Shipping Duration by Region', ylabel='Days', xlabel='Region')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'avg_ship_duration_region.png'))
plt.close()

late_deliveries = df[df['Shipping Duration'] > 5]
late_rate = len(late_deliveries) / len(df) * 100
print(f"\nLate Deliveries (>5 days): {len(late_deliveries)} orders ({late_rate:.2f}%)")

# PROFIT MARGIN ANALYSIS
pm_category = df.groupby('Category')['Profit Margin'].mean().sort_values(ascending=False)
pm_category.plot(kind='bar', title='Average Profit Margin by Category', ylabel='Profit Margin')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_margin_category.png'))
plt.close()

pm_segment = df.groupby('Segment')['Profit Margin'].mean().sort_values(ascending=False)
pm_segment.plot(kind='bar', title='Average Profit Margin by Segment', ylabel='Profit Margin')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_margin_segment.png'))
plt.close()

pm_region = df.groupby('Region')['Profit Margin'].mean().sort_values(ascending=False)
pm_region.plot(kind='bar', title='Average Profit Margin by Region', ylabel='Profit Margin')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_margin_region.png'))
plt.close()

# CUSTOMER ANALYSIS
top_customers = df.groupby('Customer ID')['Sales'].sum().nlargest(10)
top_customers.plot(kind='bar', title='Top 10 Customers by Sales', ylabel='Total Sales')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_10_customers_sales.png'))
plt.close()

avg_spending = df.groupby('Customer ID')['Sales'].mean().mean()
print(f"\nAverage Spending per Customer: {avg_spending:.2f}")

customer_stats = df.groupby('Customer ID').agg({'Sales': 'sum', 'Profit': 'mean'}).reset_index()
X = customer_stats[['Sales', 'Profit']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_stats['Cluster'] = kmeans.fit_predict(X)
sns.scatterplot(data=customer_stats, x='Sales', y='Profit', hue='Cluster', palette='Set2')
plt.title('Customer Segmentation (K-Means)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'customer_segmentation.png'))
plt.close()

customer_stats.to_csv(os.path.join(output_dir, 'customer_segmentation.csv'), index=False)

# VISUALIZATIONS
# 1. Bar chart: Sales by Region & Category
plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Region', y='Sales', hue='Category', estimator='sum')
plt.title('Sales by Region and Category')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sales_by_region_category.png'))
plt.close()

# 2. Box plot: Profit by Discount
sns.boxplot(data=df, x='Discount', y='Profit')
plt.title('Profit by Discount')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_by_discount.png'))
plt.close()

# 3. Scatter plot: Profit vs Discount
sns.scatterplot(data=df, x='Discount', y='Profit')
plt.title('Profit vs Discount')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_vs_discount.png'))
plt.close()

# 4. Correlation matrix + heatmap
corr_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Duration']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# 5. Customer segment share pie chart
segment_counts = df['Segment'].value_counts()
segment_counts.plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Customer Segment Share')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'customer_segment_pie.png'))
plt.close()

# 6. Region order count bar chart
region_order_count = df['Region'].value_counts()
region_order_count.plot(kind='bar', title='Order Count by Region')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'region_order_count.png'))
plt.close()

# Region sales bar chart
region_sales = df.groupby('Region')['Sales'].sum()
region_sales.plot(kind='bar', title='Total Sales by Region')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'region_sales.png'))
plt.close()

# 7. Monthly sales trend
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()
monthly_sales.plot(kind='line', marker='o', title='Monthly Sales Trend')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_sales_trend.png'))
plt.close()

# 8. Two-way heatmap of avg profit by Segment & Category
pivot_table = df.pivot_table(index='Segment', columns='Category', values='Profit', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Average Profit by Segment and Category')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'avg_profit_segment_category.png'))
plt.close()

print(f"All analysis complete. Charts and data saved in {output_dir}")
