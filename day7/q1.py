import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv("Dataset_Day7.csv")
temp = df.copy()

#  Identify missing values (where 0 is considered missing)
missing_cols = ["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]
MV = (temp[missing_cols] == 0).sum()

print("% of Missing values are present in:")
print((MV[MV > 0] / len(temp)) * 100)

#  Replace 0s with NaN and fill with median
for col in MV[MV > 0].index:
    temp[col] = temp[col].replace(0, np.nan)
    temp[col] = temp[col].fillna(temp[col].median())

#  Save cleaned data
df_NoMV = temp.copy()

#  Check for remaining missing values
print("\nMissing values after cleaning:")
print(df_NoMV[missing_cols].isna().sum())

#  Z-score standardization
features_to_scale = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
for col in features_to_scale:
    df_NoMV[col] = (df_NoMV[col] - df_NoMV[col].mean()) / df_NoMV[col].std()

#  Outlier detection using Z-score method
outlier_condition = (np.abs(df_NoMV[features_to_scale]) > 3).any(axis=1)
OutlierRows = df_NoMV[outlier_condition]

print("\n% of Outlier rows in the dataset is {:.2f}%\n".format(len(OutlierRows) / len(df_NoMV) * 100))

#  Remove outliers
df_NoMV_OutlierFree = df_NoMV.drop(OutlierRows.index, axis=0)

# Final dataset info
df_NoMV_OutlierFree.info()


outlier_condition = (np.abs(df_NoMV[features_to_scale]) > 3).any(axis=1)
df_NoMV_OutlierFree = df_NoMV[~outlier_condition]

#  Define input and output
X = df_NoMV_OutlierFree.drop("Outcome", axis=1)
y = df_NoMV_OutlierFree["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("✅ Q3: Data split completed (70% train, 30% test).")
print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows\n")

#  Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#  Make predictions
y_pred = model.predict(X_test)

#  Evaluate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Q3a: Model trained and evaluated.")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")