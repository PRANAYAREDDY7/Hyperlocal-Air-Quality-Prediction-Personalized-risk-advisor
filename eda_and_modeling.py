import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Load cleaned dataset
data = pd.read_csv('cleaned_air_quality_data.csv', parse_dates=['From Date'])

# --- EDA ---

print("Data Summary Statistics:")
print(data.describe())

# Plot PM2.5 distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['PM2.5 (ug/m3)'].dropna(), bins=50, kde=True)
plt.title('PM2.5 Distribution')
plt.xlabel('PM2.5 (ug/m3)')
plt.ylabel('Count')
plt.show()

# Monthly average PM2.5 trend
data['month_year'] = data['From Date'].dt.to_period('M')
monthly_avg = data.groupby('month_year')['PM2.5 (ug/m3)'].mean()
monthly_avg.plot(figsize=(14, 7), title='Monthly Average PM2.5')
plt.xlabel('Month and Year')
plt.ylabel('Average PM2.5 (ug/m3)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# --- Modeling Preparation ---

features = ['PM10 (ug/m3)', 'NO2 (ug/m3)', 'NOx (ppb)', 'SO2 (ug/m3)', 'CO (mg/m3)',
            'Ozone (ug/m3)', 'NH3 (ug/m3)', 'Temp (degree C)', 'RH (%)', 'WS (m/s)', 'RF (mm)',
            'year', 'month', 'day', 'hour', 'weekday']

# Drop rows with missing values in target or features
model_data = data.dropna(subset=['PM2.5 (ug/m3)'] + features)

X = model_data[features]
y = model_data['PM2.5 (ug/m3)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

baseline_mae = mean_absolute_error(y_test, y_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Baseline Model MAE: {baseline_mae:.2f}")
print(f"Baseline Model RMSE: {baseline_rmse:.2f}")

# --- Hyperparameter Tuning ---

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_absolute_error',
    random_state=42
)

search.fit(X_train, y_train)

print("Best hyperparameters:", search.best_params_)
print("Best CV score (negative MAE):", search.best_score_)

best_model = search.best_estimator_
y_pred_best = best_model.predict(X_test)
tuned_mae = mean_absolute_error(y_test, y_pred_best)
print(f"Tuned Model MAE: {tuned_mae:.2f}")

# --- Save the best model ---

joblib.dump(best_model, 'best_random_forest_model.joblib')
print("Best model saved as 'best_random_forest_model.joblib'")
