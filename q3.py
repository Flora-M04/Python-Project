import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("Dataset_Day7.csv")
temp = df.copy()

# Handle missing values
missing_cols = ["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]
for col in missing_cols:
    temp[col] = temp[col].replace(0, np.nan)
    temp[col] = temp[col].fillna(temp[col].median())

df_NoMV = temp.copy()

# Standardize features
features_to_scale = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
for col in features_to_scale:
    df_NoMV[col] = (df_NoMV[col] - df_NoMV[col].mean()) / df_NoMV[col].std()

# Remove outliers using Z-score
outlier_condition = (np.abs(df_NoMV[features_to_scale]) > 3).any(axis=1)
df_NoMV_OutlierFree = df_NoMV[~outlier_condition]  # Keep only non-outliers

# ✅ Now continue with Q3
X = df_NoMV_OutlierFree.drop("Outcome", axis=1)
y = df_NoMV_OutlierFree["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("✅ Q3: Data split completed (70% train, 30% test).")
print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows\n")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Q3a: Model trained and evaluated.")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
