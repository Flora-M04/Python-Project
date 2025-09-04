from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("Dataset_Day5.csv")
df = df.dropna()

X = df[['RM']]       
y = df['MEDV'] 

#Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the equation
print("Regression Equation:")
print(f"MEDV = {model.intercept_:.2f} + {model.coef_[0]:.2f} * RM")

#  R² score
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

#  Predict MEDV for RM = 7
predicted_value = model.predict([[7]])[0]
print(f"Predicted MEDV when RM = 7: {predicted_value:.2f}")
