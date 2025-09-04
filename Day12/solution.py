import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset 
data = pd.read_csv('Dataset_Day12.csv')  
print("Dataset Loaded Successfully.\n")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

missing_cols = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']

# Replace 0s with median (excluding zeros)
for col in missing_cols:
    median_val = data[data[col] != 0][col].median()
    data[col] = data[col].replace(0, median_val)
print(data[missing_cols].describe())
print("Missing data was found in four columns. These were replaced with median values, " \
"which is robust to outliers and maintains the central tendency.")

def remove_outliers_iqr(dataframe, columns):
    df_clean = dataframe.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

numeric_cols = data.columns.drop('Outcome')
df_clean = remove_outliers_iqr(data, numeric_cols)

# Show number of rows before and after outlier removal
print(f"Original dataset shape: {data.shape}")
print(f"Dataset shape after outlier removal: {df_clean.shape}")
print(f"Number of rows removed as outliers: {data.shape[0] - df_clean.shape[0]}")
print("Outliers can skew distance calculations in kNN. Removing them ensures better model stability and less variance in prediction.")

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into 80% training and 20% testing sets.\n")