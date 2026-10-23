
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data.csv")

# Input features and target
X = df[["Study_Hours", "Previous_Marks", "Attendance"]]
y = df["Final_Marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks for test records
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 45)
print("    MACHINE LEARNING MODEL EVALUATION")
print("=" * 45)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nMean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R-squared Score:", round(r2, 3))

# Save the trained model
with open("student_performance_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel trained and saved successfully!")
print("Model file: student_performance_model.pkl")

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Generate predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n" + "=" * 45)
print("    RANDOM FOREST MODEL EVALUATION")
print("=" * 45)

print("Mean Absolute Error (MAE):", round(rf_mae, 2))
print("Mean Squared Error (MSE):", round(rf_mse, 2))
print("R-squared Score:", round(rf_r2, 3))

# Save the Random Forest model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nRandom Forest model saved successfully!")