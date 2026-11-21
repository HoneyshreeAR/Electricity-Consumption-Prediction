import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load tab-separated dataset
data = pd.read_csv("electricity.csv", sep="\t")

# Display dataset information
print("Dataset Columns:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

# Select input features
X = data[["previous_usage", "appliance_power", "usage_hours"]]

# Select target output
y = data["consumption"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test results
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Training Completed!")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")