import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load and Preprocess Data
df = pd.read_csv("Dataset_Day8.csv")
temp = df.copy()

# Handle missing values: Replace 0 with NaN, then fill with median
missing_cols = ["Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]
for col in missing_cols:
    temp[col] = temp[col].replace(0, np.nan)
    temp[col] = temp[col].fillna(temp[col].median())

# Z-score normalization
features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
temp[features] = (temp[features] - temp[features].mean()) / temp[features].std()

# Remove outliers (Z-score > 3)
outlier_mask = (np.abs(temp[features]) > 3).any(axis=1)
df_cleaned = temp[~outlier_mask]

# Split Data

X = df_cleaned.drop("Outcome", axis=1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#Plot Precision & Recall vs k (Euclidean)
k_range = range(1, 31)
precisions = []
recalls = []
f1_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Identify best k
best_k = k_range[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

# Plot Precision & Recall vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_range, precisions, label='Precision', marker='o')
plt.plot(k_range, recalls, label='Recall', marker='s')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k (F1) = {best_k}')
plt.title('Precision & Recall vs k (Euclidean)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"✅ Best k (F1-Score) using Euclidean distance: k = {best_k}, F1 = {best_f1:.4f}")

#Best k and Distance Metric Combination

distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
best_combo = None
best_f1_c = 0

for metric in distance_metrics:
    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        current_f1 = f1_score(y_test, y_pred)
        if current_f1 > best_f1_c:
            best_f1_c = current_f1
            best_combo = (k, metric)

print("\n✅ Best combination for kNN:")
print(f"Best k = {best_combo[0]}")
print(f"Best distance metric = '{best_combo[1]}'")
print(f"Best F1 Score = {best_f1_c:.4f}")

# Confusion Matrix for Best Combination
best_k, best_metric = best_combo
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn_best.fit(X_train, y_train)
y_best_pred = knn_best.predict(X_test)

cm = confusion_matrix(y_test, y_best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_best.classes_)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix (k={best_k}, metric={best_metric})')
plt.show()
