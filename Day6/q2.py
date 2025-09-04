import pandas as pd

df = pd.read_csv("Dataset_Day6.csv")
print("Original shape:", df_clean.shape)
print("\nCategorical columns to encode:")
print(df_clean.select_dtypes(include='object').columns.tolist())

df_encoded = pd.get_dummies(df_clean, drop_first=True)

print("\nShape after one-hot encoding:", df_encoded.shape)
print("Columns after encoding:\n", df_encoded.columns.tolist())