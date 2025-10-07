import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # Optional extension
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model
import os

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate synthetic dataset (1000 samples) based on hydration guidelines
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(18, 80, n_samples),
    'weight': np.random.uniform(50, 120, n_samples),
    'gender': np.random.choice([0, 1], n_samples),  # 0: male, 1: female
    'activity_level': np.random.choice([0, 1, 2], n_samples),  # 0: low, 1: medium, 2: high
    'climate': np.random.choice([0, 1, 2], n_samples),  # 0: cool, 1: moderate, 2: hot
    'health': np.random.choice([0, 1], n_samples),  # 0: normal, 1: poor
    'diet_type': np.random.choice([0, 1], n_samples),  # 0: normal, 1: high-protein
    'sleep_hours': np.random.uniform(4, 10, n_samples)
}

df = pd.DataFrame(data)

# Calculate target: Base water intake (liters) with adjustments
base_intake = df['weight'] * 0.03  # 30ml/kg base
adjustments = (
    1 + (df['age'] / 100 * 0.1) +  # Age factor
    (df['activity_level'] * 0.2) +  # Activity multiplier
    (df['climate'] * 0.15) +  # Climate
    (df['health'] * -0.1) +  # Health penalty
    (df['diet_type'] * 0.1) +  # Diet
    (df['sleep_hours'] / 10 * 0.05)  # Sleep
)
df['water_intake'] = np.clip(base_intake * adjustments, 1.5, 5.0)  # Clip to realistic range (1.5-5L)

# Features and target
X = df.drop('water_intake', axis=1)
y = df['water_intake']

# Preprocessing: Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Optional: Train Random Forest for comparison (uncomment if needed)
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)  # No scaling needed for trees
# joblib.dump(rf_model, 'models/rf_water_intake_model.pkl')

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

# Save model and scaler
joblib.dump(model, 'models/water_intake_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model and scaler saved to 'models/' folder.")