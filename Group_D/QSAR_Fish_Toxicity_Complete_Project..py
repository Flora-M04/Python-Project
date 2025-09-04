{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1e7080f1",
   "metadata": {},
   "source": [
    "# 🧪 QSAR Fish Toxicity Project\n",
    "**Complete 23-Step Pipeline with Evaluation Rubric Mapping**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb0a4a28",
   "metadata": {},
   "source": [
    "\n",
    "📅 **Generated on:** July 02, 2025  \n",
    "📁 **Dataset Used:** `qsar_fish_toxicity.csv`  \n",
    "🧠 **Objective:** Predict LC50 toxicity of chemicals using machine learning with full analysis and model development pipeline  \n",
    "🎯 **Rubric Target:** Score Level 8-10 – Advanced exploration, insightful EDA, and ensemble techniques  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47e6a145",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6310ed94",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2 & 3: Upload and Explore Dataset\n",
    "df = pd.read_csv(\"qsar_fish_toxicity.csv\")\n",
    "display(df.info())\n",
    "display(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd049458",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4: Visualization - Raw Dataset\n",
    "sns.pairplot(df, diag_kind=\"hist\", plot_kws={'color': 'orange'})\n",
    "plt.suptitle(\"Pairplot of Raw Dataset\", y=1.02)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e6b8c0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 5 & 6: Data Preprocessing and Cleaning\n",
    "imputer = SimpleImputer(strategy=\"mean\")\n",
    "df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)\n",
    "print(\"Missing values filled.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b97bb097",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 7: Visualization of Cleaned Data\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')\n",
    "plt.title(\"Correlation Heatmap After Cleaning\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca36bcfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 8: PCA Analysis\n",
    "X = df_clean.drop(\"LC50 [-LOG(mol/L)]\", axis=1)\n",
    "y = df_clean[\"LC50 [-LOG(mol/L)]\"]\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "pca = PCA()\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')\n",
    "plt.xlabel('Number of Components')\n",
    "plt.ylabel('Cumulative Explained Variance')\n",
    "plt.title('PCA - Explained Variance')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58b8ec66",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 9: Train-Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "print(\" Train-test split done.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "603412df",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 10: Model Training\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Random Forest\": RandomForestRegressor(random_state=42),\n",
    "    \"SVR\": SVR(),\n",
    "    \"Gradient Boosting\": GradientBoostingRegressor(random_state=42)\n",
    "}\n",
    "trained_models = {}\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    trained_models[name] = model\n",
    "print(\"✅ Models trained.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "986d9d6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 11-13: Evaluate and Compare\n",
    "import pandas as pd\n",
    "results = {\"Model\": [], \"MSE\": [], \"R2\": []}\n",
    "for name, model in trained_models.items():\n",
    "    y_pred = model.predict(X_test)\n",
    "    results[\"Model\"].append(name)\n",
    "    results[\"MSE\"].append(mean_squared_error(y_test, y_pred))\n",
    "    results[\"R2\"].append(r2_score(y_test, y_pred))\n",
    "    print(f\"{name}: MSE={results['MSE'][-1]:.4f}, R²={results['R2'][-1]:.4f}\")\n",
    "results_df = pd.DataFrame(results)\n",
    "sns.barplot(x=\"Model\", y=\"R2\", data=results_df)\n",
    "plt.title(\"R² Scores of Models\")\n",
    "plt.show()\n",
    "sns.barplot(x=\"Model\", y=\"MSE\", data=results_df)\n",
    "plt.title(\"MSE of Models\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "487918c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 14-17: Tune Random Forest and Visualize\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 150],\n",
    "    'max_depth': [6, 8],\n",
    "    'min_samples_split': [2, 5]\n",
    "}\n",
    "grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')\n",
    "grid_search.fit(X_train, y_train)\n",
    "best_rf = grid_search.best_estimator_\n",
    "y_pred_rf = best_rf.predict(X_test)\n",
    "\n",
    "print(\"Best RF Params:\", grid_search.best_params_)\n",
    "print(\"Tuned RF - MSE:\", mean_squared_error(y_test, y_pred_rf))\n",
    "print(\"Tuned RF - R²:\", r2_score(y_test, y_pred_rf))\n",
    "\n",
    "plt.scatter(y_test, y_pred_rf, alpha=0.6)\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n",
    "plt.title(\"Actual vs Predicted (Tuned RF)\")\n",
    "plt.xlabel(\"Actual\")\n",
    "plt.ylabel(\"Predicted\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25a69f8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 18-20: Ensemble - Stacking\n",
    "stack = StackingRegressor(\n",
    "    estimators=[\n",
    "        ('lr', LinearRegression()),\n",
    "        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),\n",
    "        ('svr', SVR())\n",
    "    ],\n",
    "    final_estimator=GradientBoostingRegressor()\n",
    ")\n",
    "stack.fit(X_train, y_train)\n",
    "y_pred_stack = stack.predict(X_test)\n",
    "print(\"Stacking R²:\", r2_score(y_test, y_pred_stack))\n",
    "plt.scatter(y_test, y_pred_stack, alpha=0.6, color='purple')\n",
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n",
    "plt.title(\"Actual vs Predicted (Stacked)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6768df9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 21: Overfitting Check\n",
    "train_pred = best_rf.predict(X_train)\n",
    "print(\"Train R²:\", r2_score(y_train, train_pred))\n",
    "print(\"Test R²:\", r2_score(y_test, y_pred_rf))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
