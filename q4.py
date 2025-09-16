import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

# Define X and y
X = df_NoMV_OutlierFree.drop("Outcome", axis=1)
y = df_NoMV_OutlierFree["Outcome"]

# Split data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities for class 1
y_probs = model.predict_proba(X_test)[:, 1]

# Generate F1 scores for multiple thresholds
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_test, y_probs >= t) for t in thresholds]

# Find best threshold
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_score = max(f1_scores)

# Plot F1 Score vs Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1 Score Curve')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs. Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output
print("✅ Q4: F1 Score vs. Threshold plot completed.")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best F1 Score:  {best_f1_score:.4f}")
