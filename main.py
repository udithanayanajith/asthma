# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('diabetic_data.csv')

# Preprocess data
# Clean data, deal with missing values and outliers, and scale the features if necessary.

# Feature selection
X = data[['age', 'weight', 'blood_pressure', 'glucose_levels']]
y = data['diabetic_level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Make future predictions
age = 35
weight = 70
blood_pressure = 120
glucose_levels = 150
input_data = [[age, weight, blood_pressure, glucose_levels]]

# Predict diabetic level 5 days ahead
for i in range(5):
    prediction = model.predict(input_data)
    print(f"Diabetic level in {i+1} day(s): {prediction[0]}")
    # Update input_data with the predicted diabetic level for the next iteration
    input_data[0][-1] = prediction[0]
