import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Dataset_Day2.csv")
"""
# Get unique values(Q1)
unique_names = np.unique(df['name'])
unique_mfrs = np.unique(df['mfr'])
unique_vitamins = np.unique(df['vitamins'])

# Print results
print("Unique Names:\n", unique_names)
print("\nUnique Manufacturers:\n", unique_mfrs)
print("\nUnique Vitamins:\n", unique_vitamins)
"""


df_HighSodLowProt = df[(df['sodium'] > 100) & (df['protein'] < 3)]
"""print("\nDataframe with High Sodium and Low Protein:\n")
print(df_HighSodLowProt)
"""# Group by 'mfr' and calculate average calories
avg_calories = df_HighSodLowProt.groupby('mfr')['calories'].mean()
print("\nAverage Calories by Manufacturer:")
print(avg_calories)
# Find the manufacturer with the highest average calories
max_mfr = avg_calories.idxmax()
max_value = avg_calories.max()
print(f"\nManufacturer with highest average calories: {max_mfr} ({max_value})")



