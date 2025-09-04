import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from itertools import product

# Load dataset (replace with your actual path if needed)
data = pd.read_csv('Dataset_Day10.csv')  
print("Dataset Loaded Successfully.\n")

# Split data into features and target
 
print("Feature and Target variables split. Target variable: 'Outcome'\n")

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into 80% training and 20% testing sets.\n")

# Train Decision Tree Classifier with default parameters
clf_default = DecisionTreeClassifier(random_state=42)
clf_default.fit(X_train, y_train)
y_pred_default = clf_default.predict(X_test)
print("Default Decision Tree model trained.\n")

# Evaluate default model performance
acc = accuracy_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default)
rec = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)

print("Default Decision Tree Performance Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}\n")

# Tune hyperparameters and record performance
max_leaf_nodes_range = range(2, 20, 2)
max_depth_range = range(2, 10)

results = []
best_f1 = 0
best_params = (None, None)

for leaf_nodes, depth in product(max_leaf_nodes_range, max_depth_range):
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_leaf_nodes=leaf_nodes,
        max_depth=depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append((leaf_nodes, depth, prec, rec, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_params = (leaf_nodes, depth)

print("Hyperparameter tuning completed.")
print(f"Best F1-score: {best_f1:.4f}")
print(f"Best Parameters: max_leaf_nodes = {best_params[0]}, max_depth = {best_params[1]}\n")

# Plot Precision & Recall vs max_leaf_nodes grouped by max_depth
results_df = pd.DataFrame(results, columns=['max_leaf_nodes', 'max_depth', 'precision', 'recall', 'f1'])

plt.figure(figsize=(12, 6))
for depth in max_depth_range:
    subset = results_df[results_df['max_depth'] == depth]
    plt.plot(subset['max_leaf_nodes'], subset['precision'], label=f'Precision (depth={depth})', linestyle='--')
    plt.plot(subset['max_leaf_nodes'], subset['recall'], label=f'Recall (depth={depth})')

plt.xlabel('max_leaf_nodes')
plt.ylabel('Score')
plt.title('Precision & Recall vs max_leaf_nodes (grouped by max_depth)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Precision and Recall plot displayed successfully.\n")
