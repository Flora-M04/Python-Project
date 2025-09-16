from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
df = pd.read_csv("Dataset_Day5.csv")

# Features and target
X_dis = df [['DIS']]  # distance to employment centers
y = df ['MEDV']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dis, y, test_size=0.2, random_state=100)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Show regression equation
print("Regression Equation:")
print(f"MEDV = {model.intercept_:.2f} + {model.coef_[0]:.2f} * DIS")
# R² Score
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
# Predict MEDV for DIS = 15
predicted_value = model.predict([[15]])[0]
print(f"Predicted MEDV when DIS = 15: {predicted_value:.2f}")
