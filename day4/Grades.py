import pandas as pd
df = pd.read_csv("Dataset_Day4.csv")  

print("Missing values before treatment:")
print(df.isnull().sum())

# Fill missing numerical columns with mean or median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Confirm all missing values are handled
print("\nMissing values after treatment:")
print(df.isnull().sum())
