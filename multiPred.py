import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1: Import libraries

# Step 2: Read CSV file
data = pd.read_csv('asthmaData.csv')

# Step 3: Clean and preprocess the data
# Example: remove missing values and scale numerical variables
data = data.dropna()
data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
data['stage'] = (data['stage'] - data['stage'].mean()) / data['stage'].std()
data['pLevel'] = (data['pLevel'] - data['pLevel'].mean()) / data['pLevel'].std()
data['symIntencity'] = (data['symIntencity'] - data['symIntencity'].mean()) / data['symIntencity'].std()

# Step 4: Create a predictive model
X = data.drop(['symPerWeek'], axis=1)
y = data['symPerWeek']
model = LinearRegression().fit(X, y)

# Step 5: Use the model to make predictions
age = input("Enter age: ")
stage = input("Enter stage: ")
future_data_count = int(input("Enter the number of future data points to predict: "))
input_data = pd.DataFrame({'age': [age], 'stage': [stage]})

#input_data=[12,4,4,4]


predictions = model.predict(input_data)

# Step 6: Generate future data
future_data = []
for i in range(future_data_count):
    input_data = pd.DataFrame({'age': [age], 'stage': [stage]})
    predicted_value = model.predict(input_data)
    future_data.append(predicted_value[0])
    age += 1
    stage += 1

print("Predicted symptoms per day:", predictions[0])
print("Generated future data:", future_data)