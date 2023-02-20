import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

# Load the past data from a CSV file
past_data = pd.read_csv("asthmaData.csv")



# Train a linear regression model using the past data
reg = RandomForestClassifier().fit(past_data.drop("symPerWeek", axis=1), past_data["stage"])

# Predict the future data based on the past data
future_data = reg.predict(past_data.drop("symPerWeek", axis=1))

# Plot the past data
plt.plot(past_data["symPerWeek"], past_data["symPerWeek"], label="Past Data")

# Plot the future data
plt.plot(past_data["symPerWeek"], future_data, label="Future Data")

plt.xlabel("Symptops Per week")
plt.ylabel("Asthma Stage")

plt.legend()
plt.show()