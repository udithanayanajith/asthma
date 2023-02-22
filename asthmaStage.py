# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
import warnings
import pickle
# import os


MODEL_FILE = "asthmaModel.pkl"


warnings.filterwarnings("ignore")


def predictData(num1, num2, num3, num4):

    # print(num1,num2,num3,num4)

    # if os.path.exists(MODEL_FILE):

        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
            # Get user inputs
            inputs = [num1, num2, num3, num4]
            prediction = model.predict([inputs])[0]

            # Genarate Accurecy
            # data = pd.read_csv('asthmaData.csv')

            # X = data.drop('stage', axis=1)
            # y = data['stage']

            # # Split the data into training and testing sets
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.2, random_state=42)

            # # Create the Random Forest Classifier model
            # model = RandomForestClassifier(n_estimators=100)

            # # Train the model using the training data
            # model.fit(X_train, y_train)

            # # Make predictions using the testing data
            # y_pred = model.predict(X_test)

            # # Calculate the accuracy of the model
            # accuracy = accuracy_score(y_test, y_pred)
            # print("Accurecy:", accuracy)
            # Output the prediction
            # print("Asthma Stage:", prediction)

            return prediction

    # else:

    #     # Load the data into a Pandas DataFrame
    #     data = pd.read_csv('asthmaData.csv')

    #     X = data.drop('stage', axis=1)
    #     y = data['stage']

    #     # Split the data into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.2, random_state=42)

    #     # Create the Random Forest Classifier model
    #     model = RandomForestClassifier(n_estimators=100)

    #     # Train the model using the training data
    #     model.fit(X_train, y_train)

    #     # Make predictions using the testing data
    #     y_pred = model.predict(X_test)

    #     # Calculate the accuracy of the model
    #     # accuracy = accuracy_score(y_test, y_pred)

    #     # print("Accuracy:", accuracy)

    #     pickle.dump(model, open('asthmaModel.pkl', 'wb'))

    #     # Get user inputs
    #     inputs = [num1, num2, num3, num4]
    #     prediction = model.predict([inputs])[0]

    #     # Output the prediction
    #     # print("Asthma Stage:", prediction)

    #     return prediction
