import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_Day5.csv")

# Check for Missing Values
print("Missing Values:")
print(df.isnull().sum())
print("\n")

# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe().T)
print("\n")

# Detect Outliers using Boxplots
print("Visualizing Outliers (Boxplots):")
plt.figure(figsize=(15, 12))
for i, column in enumerate(df.select_dtypes(include='number').columns):
    plt.subplot(5, 3, i+1)
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# Histograms for Visualizing Distributions
print("Visualizing Data Distributions (Histograms):")
plt.figure(figsize=(15, 12))
for i, column in enumerate(df.select_dtypes(include='number').columns):
    plt.subplot(5, 3, i+1)
    sns.histplot(df[column], kde=True, color='orange')
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

print("\nPairplot for continuous variables:")
sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
plt.show()