import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv("Dataset_Day12.csv")

#Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a. Naive Bayes Classifier

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("\n Naive Bayes (80-20 Split)")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

# b. 10-Fold Cross Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n Naive Bayes (10-Fold Cross Validation)")
print(f"CV Accuracy:  {cross_val_score(nb, X, y, cv=cv, scoring='accuracy').mean():.3f}")
print(f"CV Precision: {cross_val_score(nb, X, y, cv=cv, scoring='precision').mean():.3f}")
print(f"CV Recall:    {cross_val_score(nb, X, y, cv=cv, scoring='recall').mean():.3f}")
print(f"CV F1 Score:  {cross_val_score(nb, X, y, cv=cv, scoring='f1').mean():.3f}")

