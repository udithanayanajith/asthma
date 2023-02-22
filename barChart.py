import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def genarateBarChart(plotData):
    print(plotData, "Data inside the barchart script")
    data = pd.read_csv('asthmaData.csv')

    X = data.drop('symPerWeek', axis=1)
    y = data['symPerWeek']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Choose and train a machine learning model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    inputs = np.array(plotData)
    # inputs = np.array([[19,23,98,1]])

    # Make predictions based on user inputs
    y_pred = model.predict(inputs)

    # Print the predicted target variable for the next 5 houses
    print(y_pred)
    prediction_array = np.array(y_pred)
    # convert to a Python list
    prediction_list = prediction_array.tolist()

    print(prediction_list,"pred list")
    # last_four = data.tail(4)

    # # Get the values of the last four data points
    # values = last_four['symPerWeek'].values

    # labels = ['PDay1', 'PDay2', 'PDay3', 'PDay4',
    #           'FDay1', 'FDay2', 'FDay3', 'FDay4']

    # y_values = list(values) + list(y_pred)

    # colors = ['b', 'b', 'b', 'b', 'r', 'r', 'r', 'r']

    # Plot the data as a bar chart
    # plt.bar(labels, y_values, color=colors)
    # plt.xlabel('Days')
    # plt.ylabel('Asthma Symptoms PerWeek')
    # plt.legend()
    # plt.show()

    return prediction_list
