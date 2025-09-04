import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv('Dataset_Day4.csv')  # Replace with actual filename
# Select numeric columns only
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# --- Boxplot & Outlier Detection ---
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna(), vert=True)
    plt.title(f'Boxplot for {col}')
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

    # Outlier detection using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if not outliers.empty:
        print(f"\nOutliers in '{col}':")
        print(outliers[[col]])
    else:
        print(f"No outliers in '{col}'\n")
# --- Correlation Matrix ---
correlation_matrix = df[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot the correlation matrix using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
