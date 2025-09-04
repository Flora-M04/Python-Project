# Data manipulation and processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score

# Regression models
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Warnings
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"qsar_fish_toxicity.csv")
#Display the first five rows:
data.head()

#Basic structure of the dataset
print("Shape of Dataset:",data.shape)
print("\nColumn Names:\n",data.columns)

# Checking data types and nulls
data.info()

data.describe().transpose()

# Check for missing values
print("Missing values:\n",data.isnull().sum())
# Check for zeros in SM1_Dz(Z)
zero_count=(data["SM1_Dz(Z)"] == 0).sum()
print(f"\nNumber of zero values in 'SM1_Dz(Z)':{zero_count}")

# Heatmap of missing values
plt.figure(figsize=(8,4))
sns.heatmap(data.isnull(), cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()

# Distribution of all features
data.hist(bins=30, figsize=(12,8), edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Replace 0s in SM1_Dz(Z) with NaN
data["SM1_Dz(Z)"]=data["SM1_Dz(Z)"].replace(0,np.nan)
# Replace SM1_Dz(Z) NaNs with median of valid values
sm1_median=data["SM1_Dz(Z)"].median()
data["SM1_Dz(Z)"].fillna(sm1_median, inplace=True)
# Fill other numeric NaNs with median
data.fillna(data.median(numeric_only=True),inplace=True)
# Confirm no missing values remain
print("Missing values after cleaning:\n", data.isnull().sum())

# Detect Outliers
outlier_df = pd.DataFrame()
for col in data.select_dtypes(include=np.number).columns:
    Q1=data[col].quantile(0.25)
    Q3=data[col].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outlier_df[col]=data[col][(data[col]<lower_bound)|(data[col]>upper_bound)]

# Show which rows had outliers
display(outlier_df)

# Removal of Outlier Rows
data_cleaned = data.drop(outlier_df.index, axis=0)

# Checking of the new shape
print(f"Original shape:{data.shape}")
print(f"After outlier removal:{data_cleaned.shape}")

plt.figure(figsize=(12, 6))
sns.boxplot(data=data_cleaned)
plt.title("Boxplot After Outlier Removal")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Separate features and target
X=data_cleaned.drop("LC50 [-LOG(mol/L)]",axis=1)
y=data_cleaned["LC50 [-LOG(mol/L)]"]
# Scale the features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
# Train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

#Train and Predict
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
# Metrics
lr_r2 = r2_score(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, y_pred_lr)

# Print results
print("Linear Regression Performance:")
print(f"R² Score:{lr_r2:.4f}")
print(f"MSE:{lr_mse:.4f}")
print(f"RMSE:{lr_rmse:.4f}")
print(f"MAE:{lr_mae:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_lr, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - y_pred_lr
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='skyblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Linear Regression")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train & Predict
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
# Metrics
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)

# Print results
print("Ridge Regression Performance:")
print(f"R² Score:{ridge_r2:.4f}")
print(f"MSE:{ridge_mse:.4f}")
print(f"RMSE:{ridge_rmse:.4f}")
print(f"MAE:{ridge_mae:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_ridge, color='darkorange', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Ridge Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual Plot
residuals_ridge = y_test - y_pred_ridge
plt.figure(figsize=(6, 4))
sns.histplot(residuals_ridge, kde=True, color='salmon')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Ridge Regression")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train & Predict
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
# Metrics
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

# Print results
print("Lasso Regression Performance:")
print(f"R² Score:{lasso_r2:.4f}")
print(f"MSE:{lasso_mse:.4f}")
print(f"RMSE:{lasso_rmse:.4f}")
print(f"MAE:{lasso_mae:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_lasso, color='blueviolet', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Lasso Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual Plot
residuals_lasso = y_test - y_pred_lasso
plt.figure(figsize=(6, 4))
sns.histplot(residuals_lasso, kde=True, color='plum')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Lasso Regression")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train & Predict
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Metrics
dt_r2 = r2_score(y_test, y_pred_dt)
dt_mse = mean_squared_error(y_test, y_pred_dt)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, y_pred_dt)

# Print results
print("Decision Tree Regression Performance:")
print(f"R² Score:{dt_r2:.4f}")
print(f"MSE:{dt_mse:.4f}")
print(f"RMSE:{dt_rmse:.4f}")
print(f"MAE:{dt_mae:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_dt, color='dodgerblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Decision Tree: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual Plot
residuals_dt = y_test - y_pred_dt
plt.figure(figsize=(6, 4))
sns.histplot(residuals_dt, kde=True, color='lightblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Decision Tree")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train & Predict
svr=SVR()
svr.fit(X_train,y_train)
y_pred_svr=svr.predict(X_test)

# Metrics
svr_r2=r2_score(y_test, y_pred_svr)
svr_mse=mean_squared_error(y_test,y_pred_svr)
svr_rmse=np.sqrt(svr_mse)
svr_mae=mean_absolute_error(y_test,y_pred_svr)

# Print results
print("SVR Performance:")
print(f"R² Score:{svr_r2:.4f}")
print(f"MSE:{svr_mse:.4f}")
print(f"RMSE:{svr_rmse:.4f}")
print(f"MAE:{svr_mae:.4f}")


# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_svr, color='darkgreen', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("SVR: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual Plot
residuals_svr = y_test - y_pred_svr
plt.figure(figsize=(6, 4))
sns.histplot(residuals_svr, kde=True, color='mediumseagreen')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - SVR")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Create model comparison DataFrame
results_df=pd.DataFrame({
    "Model": [
        "Linear Regression", 
        "Ridge Regression", 
        "Lasso Regression", 
        "Decision Tree", 
        "SVR"
    ],
    "R² Score": [
        lr_r2, 
        ridge_r2, 
        lasso_r2, 
        dt_r2, 
        svr_r2
    ],
    "MAE": [
        lr_mae, 
        ridge_mae, 
        lasso_mae, 
        dt_mae, 
        svr_mae
    ],
    "RMSE": [
        lr_rmse, 
        ridge_rmse, 
        lasso_rmse, 
        dt_rmse, 
        svr_rmse
    ]
})

# R² Score Comparison
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df,x="Model",y="R² Score",palette="Blues_d")
plt.title(" Model Comparison - R² Score", fontsize=14)
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.grid(True,linestyle='--',alpha=0.6)
plt.tight_layout()
plt.show()

# MAE and RMSE Comparison
fig,ax=plt.subplots(1, 2, figsize=(14, 5))

# MAE
sns.barplot(data=results_df,x="Model",y="MAE",ax=ax[0],palette="Greens_d")
ax[0].set_title("Model Comparison - MAE",fontsize=13)
ax[0].set_ylabel("MAE")
ax[0].tick_params(axis='x',rotation=45)
ax[0].grid(True, linestyle='--',alpha=0.6)

# RMSE
sns.barplot(data=results_df, x="Model",y="RMSE",ax=ax[1],palette="Reds_d")
ax[1].set_title("Model Comparison - RMSE", fontsize=13)
ax[1].set_ylabel("RMSE")
ax[1].tick_params(axis='x', rotation=45)
ax[1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle("Model Performance Comparison", fontsize=15, y=1.05)
plt.tight_layout()
plt.show()

alphas=[0.01, 0.1, 1, 10, 100]
print("Ridge Regression Hyperparameter Tuning\n")
for alpha in alphas:
    print(f"Evaluating Ridge with Alpha = {alpha}")
    # Train
    model = Ridge(alpha=alpha)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # Metrics
    r2=r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    # Print metrics
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Actual vs Predicted Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, color='orange',alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual LC50")
    plt.ylabel("Predicted LC50")
    plt.title(f"Ridge Regression (Alpha = {alpha}) - Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color='goldenrod')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Residuals Distribution - Ridge (α = {alpha})")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.show()

    print("-" * 60)

depths=[2, 4, 6, 8]
min_leafs=[1, 2, 4]
print("Decision Tree Tuning (with min_samples_leaf)\n")
for d in depths:
    for m in min_leafs:
        print(f"Max Depth = {d}, Min Samples Leaf = {m}")
        # Train the model
        model = DecisionTreeRegressor(max_depth=d, min_samples_leaf=m, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Print metrics
        print(f"R² Score:{r2:.4f}")
        print(f"MAE:{mae:.4f}")
        print(f"RMSE:{rmse:.4f}")

        # Actual vs Predicted Plot
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.xlabel("Actual LC50")
        plt.ylabel("Predicted LC50")
        plt.title(f"Decision Tree (Depth={d}, Leaf={m}) - Actual vs Predicted")
        plt.tight_layout()
        plt.show()

        # Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, color='lightskyblue')
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f"Residuals Distribution - DT (Depth={d}, Leaf={m})")
        plt.xlabel("Residuals")
        plt.tight_layout()
        plt.show()

        print("-" * 60)
Cs=[0.1, 1, 10, 100]
print("SVR Hyperparameter Tuning (kernel = 'rbf')\n")
for c in Cs:
    print(f"Evaluating SVR with C = {c}")
    # Train
    model = SVR(C=c, kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Print results
    print(f"R² Score:{r2:.4f}")
    print(f"MAE:{mae:.4f}")
    print(f"RMSE:{rmse:.4f}")

    # Actual vs Predicted Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred, color='mediumseagreen', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual LC50")
    plt.ylabel("Predicted LC50")
    plt.title(f"SVR (C = {c}) - Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color='seagreen')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Residuals Distribution - SVR (C = {c})")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.show()

    print("-" * 60)
# Train best Ridge model
best_ridge = Ridge(alpha=100)
best_ridge.fit(X_train, y_train)
y_pred = best_ridge.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Ridge Model (Alpha = 100)")
print(f"R² Score:{r2:.4f}")
print(f"MAE:{mae:.4f}")
print(f"RMSE:{rmse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color='darkblue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Ridge (Best Alpha = 100) - Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='navy')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Ridge (Alpha = 100)")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train best Decision Tree model
best_dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42)
best_dt.fit(X_train, y_train)
y_pred = best_dt.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" Best Decision Tree (Depth = 6, Leaf = 4)")
print(f"R² Score:{r2:.4f}")
print(f"MAE:{mae:.4f}")
print(f"RMSE:{rmse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color='darkgreen', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Decision Tree (Best Depth = 6, Leaf = 4) - Actual vs Predicted")
plt.tight_layout()
plt.show()

#  Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='seagreen')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Decision Tree (Depth = 6, Leaf = 4)")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train best SVR model
best_svr = SVR(C=10, kernel='rbf')
best_svr.fit(X_train, y_train)
y_pred = best_svr.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" Best SVR Model (C = 10, Kernel = 'rbf')")
print(f"R² Score:{r2:.4f}")
print(f"MAE:{mae:.4f}")
print(f"RMSE:{rmse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("SVR (Best C = 10) - Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='orchid')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - SVR (C = 10)")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" Random Forest Regressor")
print(f"R² Score:{r2:.4f}")
print(f"MAE:{mae:.4f}")
print(f"RMSE:{rmse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color='forestgreen', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("Random Forest - Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='mediumseagreen')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()


# Train AdaBoost
ada = AdaBoostRegressor(random_state=42)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" AdaBoost Regressor")
print(f"R² Score:{r2:.4f}")
print(f"MAE:{mae:.4f}")
print(f"RMSE:{rmse:.4f}")

#  Actual vs Predicted Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color='darkred', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual LC50")
plt.ylabel("Predicted LC50")
plt.title("AdaBoost - Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='indianred')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution - AdaBoost")
plt.xlabel("Residuals")
plt.tight_layout()
plt.show()

# Clearing previous results
final_results = []
# Evaluating all best models again and storing their scores
def store_final_result(model,name):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    final_results.append({
        "Model": name,
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse
    })

# Store results for ensemble models
store_final_result(bagging,"Bagging Regressor")
store_final_result(rf,"Random Forest")
store_final_result(ada,"AdaBoost")

ensemble_df=pd.DataFrame(final_results[-3:])
combined_df=pd.DataFrame(final_results)    

# Plot R² for all models
plt.figure(figsize=(10, 5))
sns.barplot(data=combined_df, x="Model", y="R²", palette="coolwarm")
plt.title("Final Comparison - R² Score (All Models)")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# 5-Fold Cross-Validation on Random Forest
rf_cv=RandomForestRegressor(random_state=42)
cv_scores=cross_val_score(rf_cv,X,y,cv=5,scoring='r2')
print("Cross-Validation R² Scores:",cv_scores)
print("Average CV Score:",np.mean(cv_scores).round(4))

sns.barplot(x=[f'Fold {i+1}' for i in range(len(cv_scores))], y=cv_scores, palette='viridis')
plt.title(" Cross-Validation R² Scores per Fold")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# Store results in a list of dictionaries
model_results=[]
# Linear Regression
model_results.append({
    'Model': 'Linear Regression',
    'R2': lr_r2,
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
})
# Ridge
model_results.append({
    'Model': 'Ridge',
    'R2': ridge_r2,
    'MAE': mean_absolute_error(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge))
})
# Lasso
model_results.append({
    'Model': 'Lasso',
    'R2': lasso_r2,
    'MAE': mean_absolute_error(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso))
})
# Decision Tree
model_results.append({
    'Model': 'Decision Tree',
    'R2': dt_r2,
    'MAE': mean_absolute_error(y_test, y_pred_dt),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt))
})
# SVR
model_results.append({
    'Model': 'SVR',
    'R2': svr_r2,
    'MAE': mean_absolute_error(y_test, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr))
})
# Bagging
model_results.append({
    'Model': 'Bagging',
    'R2': r2_score(y_test, bagging.predict(X_test)),
    'MAE': mean_absolute_error(y_test, bagging.predict(X_test)),
    'RMSE': np.sqrt(mean_squared_error(y_test, bagging.predict(X_test)))
})
# Random Forest
model_results.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test, rf.predict(X_test)),
    'MAE': mean_absolute_error(y_test, rf.predict(X_test)),
    'RMSE': np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
})
# AdaBoost
model_results.append({
    'Model': 'AdaBoost',
    'R2': r2_score(y_test, ada.predict(X_test)),
    'MAE': mean_absolute_error(y_test, ada.predict(X_test)),
    'RMSE': np.sqrt(mean_squared_error(y_test, ada.predict(X_test)))
})
# Convert to DataFrame
results_df = pd.DataFrame(model_results)
results_df.sort_values(by='R2', ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)
# Show Table
print(" Model Performance Comparison:")
display(results_df.style.background_gradient(cmap='YlGnBu'))

# Plot R² Score
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='R2', palette='Blues_d')
plt.title("Model Comparison - R² Score", fontsize=14)
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot MAE and RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=results_df, x='Model', y='MAE', palette='Greens_d', ax=ax[0])
ax[0].set_title("Model Comparison - MAE", fontsize=12)
ax[0].tick_params(axis='x', rotation=45)
ax[0].grid(True)

sns.barplot(data=results_df, x='Model', y='RMSE', palette='Reds_d', ax=ax[1])
ax[1].set_title("Model Comparison - RMSE", fontsize=12)
ax[1].tick_params(axis='x', rotation=45)
ax[1].grid(True)

plt.tight_layout()
plt.show()
